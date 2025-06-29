#!/usr/bin/env python3
"""
Advanced Variable Load Test for Aggressive HPA Training Data Collection
Generates multiple load patterns to trigger frequent scaling events
"""

from locust import HttpUser, task, between, events
import random
import time
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VariableLoadUser(HttpUser):
    wait_time = between(0.5, 2.0)  # Variable wait time between requests
    
    def on_start(self):
        """Called when a user starts"""
        self.session_start_time = time.time()
        logger.info(f"User {self.environment.runner.user_count} started")
    
    @task(3)
    def browse_homepage(self):
        """Main homepage browsing - highest frequency"""
        self.client.get("/")
    
    @task(2)
    def browse_products(self):
        """Browse product catalog"""
        # Get product list first
        response = self.client.get("/")
        if response.status_code == 200:
            # Try to browse a specific product
            product_ids = [
                "OLJCESPC7Z", "66VCHSJNUP", "1YMWWN1N4O", 
                "L9ECAV7KIM", "2ZYFJ3GM2N", "0PUK6V6EV0"
            ]
            product_id = random.choice(product_ids)
            self.client.get(f"/product/{product_id}")
    
    @task(1)
    def add_to_cart_and_checkout(self):
        """Complex operation - cart and checkout simulation"""
        # Browse product
        product_ids = ["OLJCESPC7Z", "66VCHSJNUP", "1YMWWN1N4O"]
        product_id = random.choice(product_ids)
        
        # View product
        self.client.get(f"/product/{product_id}")
        
        # Add to cart
        self.client.post("/cart", data={
            "product_id": product_id,
            "quantity": random.randint(1, 3)
        })
        
        # View cart
        self.client.get("/cart")
        
        # Sometimes proceed to checkout (resource intensive)
        if random.random() < 0.3:
            self.client.post("/cart/checkout", data={
                "email": f"test{random.randint(1, 1000)}@example.com",
                "street_address": "123 Test St",
                "zip_code": "12345",
                "city": "TestCity",
                "state": "TestState",
                "country": "US",
                "credit_card_number": "4111111111111111",
                "credit_card_expiration_month": "12",
                "credit_card_expiration_year": "2025",
                "credit_card_cvv": "123"
            })

# Custom load shape for variable load patterns
class VariableLoadShape:
    """
    Generates different load patterns throughout the test
    """
    
    def __init__(self):
        self.start_time = time.time()
        
    def get_current_load(self):
        """Calculate current load based on time and pattern"""
        elapsed = time.time() - self.start_time
        
        # Different phases of the test
        if elapsed < 120:  # First 2 minutes: gradual ramp up
            return int(10 + (elapsed / 120) * 40), 2
        elif elapsed < 240:  # Next 2 minutes: oscillating load
            base_load = 50
            oscillation = 30 * math.sin(elapsed * 0.1)
            return int(base_load + oscillation), 3
        elif elapsed < 360:  # Next 2 minutes: spike pattern
            if int(elapsed) % 60 < 20:  # 20 seconds high, 40 seconds low
                return 100, 5  # High spike
            else:
                return 20, 1   # Low baseline
        elif elapsed < 480:  # Next 2 minutes: random spikes
            if random.random() < 0.2:  # 20% chance of spike
                return random.randint(80, 120), 4
            else:
                return random.randint(10, 40), 2
        elif elapsed < 600:  # Final 2 minutes: sustained high load
            return 80, 4
        else:  # Gradual ramp down
            return max(10, int(80 - ((elapsed - 600) / 120) * 70)), 1

# Event handlers for better monitoring
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    logger.info("=== Advanced Variable Load Test Started ===")
    logger.info("This test will generate multiple load patterns over ~12 minutes")
    logger.info("Monitor HPA with: watch -n 2 'kubectl get hpa'")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    logger.info("=== Advanced Variable Load Test Completed ===")
    logger.info("Check your metrics collection and HPA scaling events")

# Custom user spawning for realistic patterns
class RealisticSpawning(HttpUser):
    """User with more realistic behavior patterns"""
    wait_time = between(1, 5)
    
    def on_start(self):
        self.user_type = random.choice(['browser', 'shopper', 'power_user'])
        self.session_duration = random.randint(30, 300)  # 30 seconds to 5 minutes
        self.start_time = time.time()
    
    @task
    def user_behavior(self):
        """Behavior based on user type"""
        if time.time() - self.start_time > self.session_duration:
            self.environment.runner.quit()
            return
            
        if self.user_type == 'browser':
            # Light browsing
            if random.random() < 0.7:
                self.client.get("/")
            else:
                product_id = random.choice(["OLJCESPC7Z", "66VCHSJNUP"])
                self.client.get(f"/product/{product_id}")
                
        elif self.user_type == 'shopper':
            # Active shopping
            if random.random() < 0.4:
                self.client.get("/")
            elif random.random() < 0.7:
                product_id = random.choice(["OLJCESPC7Z", "66VCHSJNUP", "1YMWWN1N4O"])
                self.client.get(f"/product/{product_id}")
                # Add to cart sometimes
                if random.random() < 0.3:
                    self.client.post("/cart", data={
                        "product_id": product_id,
                        "quantity": 1
                    })
            else:
                self.client.get("/cart")
                
        else:  # power_user
            # Heavy usage
            actions = [
                lambda: self.client.get("/"),
                lambda: self.client.get(f"/product/{random.choice(['OLJCESPC7Z', '66VCHSJNUP', '1YMWWN1N4O', 'L9ECAV7KIM'])}"),
                lambda: self.client.get("/cart"),
                lambda: self.client.post("/cart", data={"product_id": "OLJCESPC7Z", "quantity": 1})
            ]
            # Execute multiple actions
            for _ in range(random.randint(1, 3)):
                random.choice(actions)()
                time.sleep(random.uniform(0.1, 0.5))
