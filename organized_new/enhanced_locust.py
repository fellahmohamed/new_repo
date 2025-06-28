import random
import time
from locust import HttpUser, task, between, events
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AggressiveLoadUser(HttpUser):
    """
    Aggressive user behavior designed to trigger rapid HPA scaling
    within 3 minutes with multiple scale up/down cycles
    """
    
    wait_time = between(0.1, 0.5)  # Very short wait time for aggressive load
    
    def on_start(self):
        """Initialize user session"""
        self.product_ids = []
        self.session_id = f"session_{random.randint(1000, 9999)}"
        self.currency_codes = ["USD", "EUR", "CAD", "JPY", "GBP", "AUD", "CHF", "CNY"]
        self.discover_products()
        
        # Track test phase for variable load
        self.start_time = time.time()
    
    def get_test_phase(self):
        """Determine current test phase for variable load patterns"""
        elapsed = time.time() - self.start_time
        if elapsed < 30:
            return "warmup"
        elif elapsed < 60:
            return "spike1"
        elif elapsed < 90:
            return "cooldown1"
        elif elapsed < 120:
            return "spike2"
        elif elapsed < 150:
            return "cooldown2"
        else:
            return "finale"
    
    def discover_products(self):
        """Get available products to use in other tasks"""
        try:
            response = self.client.get("/")
            if response.status_code == 200:
                self.product_ids = [
                    "0PUK6V6EV0", "1YMWWN1N4O", "2ZYFJ3GM2N", "66VCHSJNUP", 
                    "6E92ZMYYFZ", "9SIQT8TOJO", "L9ECAV7KIM", "LS4PSXUNUM", "OLJCESPC7Z"
                ]
                logger.info(f"Loaded {len(self.product_ids)} products")
        except Exception as e:
            logger.error(f"Failed to discover products: {e}")
            self.product_ids = ["0PUK6V6EV0", "1YMWWN1N4O", "2ZYFJ3GM2N"]
    
    @task(20)
    def aggressive_homepage_browsing(self):
        """Aggressive homepage browsing with currency changes"""
        phase = self.get_test_phase()
        
        # Variable request frequency based on phase
        if phase in ["spike1", "spike2", "finale"]:
            # High load phases - multiple rapid requests
            for i in range(random.randint(2, 4)):
                currency = random.choice(self.currency_codes)
                with self.client.get(f"/?currency_code={currency}", 
                                   catch_response=True, 
                                   name=f"homepage_spike_{phase}") as response:
                    if response.status_code != 200:
                        response.failure(f"Homepage failed: {response.status_code}")
                time.sleep(0.1)  # Very short delay between requests
        else:
            # Normal load
            with self.client.get("/", catch_response=True, name="homepage_normal") as response:
                if response.status_code != 200:
                    response.failure(f"Homepage failed: {response.status_code}")
    
    @task(25)
    def intensive_product_browsing(self):
        """Intensive product browsing with rapid currency switching"""
        phase = self.get_test_phase()
        
        if phase in ["spike1", "spike2"]:
            # Rapid fire requests during spike phases
            for _ in range(random.randint(3, 6)):
                currency = random.choice(self.currency_codes)
                with self.client.get(f"/?currency_code={currency}", 
                                   catch_response=True, 
                                   name=f"product_browsing_spike") as response:
                    if response.status_code != 200:
                        response.failure(f"Product browsing failed: {response.status_code}")
                time.sleep(0.05)  # Minimal delay
        else:
            currency = random.choice(self.currency_codes)
            with self.client.get(f"/?currency_code={currency}", 
                               catch_response=True, 
                               name="product_browsing_normal") as response:
                if response.status_code != 200:
                    response.failure(f"Product browsing failed: {response.status_code}")
    
    @task(18)
    def rapid_product_views(self):
        """Rapidly view multiple product details"""
        phase = self.get_test_phase()
        
        if not self.product_ids:
            return
            
        if phase in ["spike1", "spike2", "finale"]:
            # View multiple products rapidly
            for _ in range(random.randint(2, 5)):
                product_id = random.choice(self.product_ids)
                currency = random.choice(self.currency_codes)
                with self.client.get(f"/product/{product_id}?currency_code={currency}",
                                   catch_response=True,
                                   name=f"product_detail_spike") as response:
                    if response.status_code != 200:
                        response.failure(f"Product detail failed: {response.status_code}")
                time.sleep(0.1)
        else:
            product_id = random.choice(self.product_ids)
            currency = random.choice(self.currency_codes)
            with self.client.get(f"/product/{product_id}?currency_code={currency}",
                               catch_response=True,
                               name="product_detail_normal") as response:
                if response.status_code != 200:
                    response.failure(f"Product detail failed: {response.status_code}")
    
    @task(15)
    def aggressive_cart_operations(self):
        """Aggressive cart add/remove operations"""
        phase = self.get_test_phase()
        
        if not self.product_ids:
            return
            
        if phase in ["spike1", "spike2"]:
            # Rapid cart operations
            for _ in range(random.randint(2, 4)):
                product_id = random.choice(self.product_ids)
                quantity = random.randint(1, 3)
                currency = random.choice(self.currency_codes)
                
                # Add to cart
                with self.client.post("/cart", 
                                    data={
                                        "product_id": product_id,
                                        "quantity": quantity,
                                        "currency_code": currency
                                    },
                                    catch_response=True,
                                    name="cart_add_spike") as response:
                    if response.status_code not in [200, 302]:
                        response.failure(f"Cart add failed: {response.status_code}")
                
                time.sleep(0.1)
        else:
            # Normal cart operation
            product_id = random.choice(self.product_ids)
            quantity = random.randint(1, 2)
            currency = random.choice(self.currency_codes)
            
            with self.client.post("/cart", 
                                data={
                                    "product_id": product_id,
                                    "quantity": quantity,
                                    "currency_code": currency
                                },
                                catch_response=True,
                                name="cart_add_normal") as response:
                if response.status_code not in [200, 302]:
                    response.failure(f"Cart add failed: {response.status_code}")
    
    @task(8)
    def view_cart_repeatedly(self):
        """View cart multiple times to generate load"""
        phase = self.get_test_phase()
        
        repeat_count = 3 if phase in ["spike1", "spike2", "finale"] else 1
        
        for _ in range(repeat_count):
            with self.client.get("/cart", catch_response=True, name="view_cart") as response:
                if response.status_code != 200:
                    response.failure(f"Cart view failed: {response.status_code}")
            if repeat_count > 1:
                time.sleep(0.1)
    
    @task(5)
    def memory_intensive_search(self):
        """Perform searches that might be memory intensive"""
        phase = self.get_test_phase()
        search_terms = ["shirt", "shoes", "jacket", "hat", "pants", "dress", "accessories"]
        
        if phase in ["spike1", "spike2"]:
            # Multiple searches rapidly
            for _ in range(random.randint(2, 4)):
                term = random.choice(search_terms)
                with self.client.get(f"/?q={term}", 
                                   catch_response=True,
                                   name="search_spike") as response:
                    if response.status_code != 200:
                        response.failure(f"Search failed: {response.status_code}")
                time.sleep(0.1)
        else:
            term = random.choice(search_terms)
            with self.client.get(f"/?q={term}", 
                               catch_response=True,
                               name="search_normal") as response:
                if response.status_code != 200:
                    response.failure(f"Search failed: {response.status_code}")

# Event listener to log phase changes
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    logger.info("🚀 Starting aggressive load test - expect rapid HPA scaling!")

@events.test_stop.add_listener  
def on_test_stop(environment, **kwargs):
    logger.info("🏁 Aggressive load test completed")
