import random
import time
from locust import HttpUser, task, between, events
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OnlineBoutiqueUser(HttpUser):
    """
    Enhanced user behavior that generates more CPU-intensive load
    to trigger HPA scaling
    """
    
    wait_time = between(1, 3)  # Shorter wait time for higher load
    
    def on_start(self):
        """Initialize user session"""
        self.product_ids = []
        self.session_id = f"session_{random.randint(1000, 9999)}"
        self.currency_codes = ["USD", "EUR", "CAD", "JPY", "GBP"]
        self.discover_products()
    
    def discover_products(self):
        """Get available products to use in other tasks"""
        try:
            response = self.client.get("/")
            if response.status_code == 200:
                # Simple product ID extraction - adjust based on actual HTML structure
                self.product_ids = [
                    "0PUK6V6EV0", "1YMWWN1N4O", "2ZYFJ3GM2N", "66VCHSJNUP", 
                    "6E92ZMYYFZ", "9SIQT8TOJO", "L9ECAV7KIM", "LS4PSXUNUM", "OLJCESPC7Z"
                ]
                logger.info(f"Loaded {len(self.product_ids)} products")
        except Exception as e:
            logger.error(f"Failed to discover products: {e}")
            # Fallback product IDs
            self.product_ids = ["0PUK6V6EV0", "1YMWWN1N4O", "2ZYFJ3GM2N"]
    
    @task(10)
    def browse_homepage(self):
        """Browse homepage - high frequency task"""
        with self.client.get("/", catch_response=True, name="homepage") as response:
            if response.status_code != 200:
                response.failure(f"Homepage failed with status {response.status_code}")
    
    @task(15)
    def browse_products(self):
        """Browse product catalog with different currencies"""
        currency = random.choice(self.currency_codes)
        with self.client.get(f"/?currency_code={currency}", 
                           catch_response=True, 
                           name="browse_products") as response:
            if response.status_code != 200:
                response.failure(f"Product browsing failed with status {response.status_code}")
    
    @task(12)
    def view_product_details(self):
        """View individual product details - CPU intensive"""
        if not self.product_ids:
            return
            
        product_id = random.choice(self.product_ids)
        currency = random.choice(self.currency_codes)
        
        with self.client.get(f"/product/{product_id}?currency_code={currency}", 
                           catch_response=True, 
                           name="product_details") as response:
            if response.status_code != 200:
                response.failure(f"Product details failed with status {response.status_code}")

class HighLoadUser(OnlineBoutiqueUser):
    """
    User class for generating very high load to definitely trigger HPA
    """
    
    wait_time = between(0.1, 0.5)  # Much shorter wait time
    
    @task(20)
    def rapid_fire_requests(self):
        """Make rapid requests to increase CPU load"""
        # Make multiple concurrent-like requests
        endpoints = ["/", "/cart"]
        
        if self.product_ids:
            endpoints.extend([f"/product/{pid}" for pid in self.product_ids[:3]])
        
        for endpoint in random.sample(endpoints, min(3, len(endpoints))):
            self.client.get(endpoint)
