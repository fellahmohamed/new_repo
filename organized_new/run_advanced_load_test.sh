#!/bin/bash

# Advanced Variable Load Test Runner
# Runs the advanced load test with different patterns

FRONTEND_URL="http://$(minikube ip):$(kubectl get service frontend-external -o jsonpath='{.spec.ports[0].nodePort}')"
METRICS_DIR="/home/mohamed/Desktop/Pfe_new/organized_new/metrics_data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_NAME="aggressive_hpa_test_${TIMESTAMP}"

echo "=== Advanced Variable Load Test for Aggressive HPA ==="
echo "Frontend URL: $FRONTEND_URL"
echo "Test Name: $TEST_NAME"
echo "Metrics Directory: $METRICS_DIR"

# Ensure metrics directory exists
mkdir -p "$METRICS_DIR"

# Function to check if frontend is accessible
check_frontend() {
    echo "Checking frontend accessibility..."
    if curl -s --max-time 10 "$FRONTEND_URL" > /dev/null; then
        echo "✓ Frontend is accessible at $FRONTEND_URL"
        return 0
    else
        echo "✗ Frontend is not accessible. Please check your deployment."
        echo "Run: kubectl get svc frontend-external"
        return 1
    fi
}

# Function to start metrics collection
start_metrics_collection() {
    echo "Starting metrics collection..."
    
    # Start HPA monitoring
    kubectl get hpa > "$METRICS_DIR/${TEST_NAME}_hpa_initial.txt"
    
    # Start resource monitoring in background
    {
        while true; do
            echo "=== $(date) ===" >> "$METRICS_DIR/${TEST_NAME}_resources.log"
            kubectl top pods >> "$METRICS_DIR/${TEST_NAME}_resources.log" 2>/dev/null
            kubectl get hpa >> "$METRICS_DIR/${TEST_NAME}_hpa_log.txt"
            echo "" >> "$METRICS_DIR/${TEST_NAME}_resources.log"
            sleep 10
        done
    } &
    METRICS_PID=$!
    echo "Metrics collection started (PID: $METRICS_PID)"
    echo $METRICS_PID > "$METRICS_DIR/${TEST_NAME}_metrics.pid"
}

# Function to stop metrics collection
stop_metrics_collection() {
    if [ -f "$METRICS_DIR/${TEST_NAME}_metrics.pid" ]; then
        METRICS_PID=$(cat "$METRICS_DIR/${TEST_NAME}_metrics.pid")
        kill $METRICS_PID 2>/dev/null
        rm "$METRICS_DIR/${TEST_NAME}_metrics.pid"
        echo "Metrics collection stopped"
    fi
    
    # Final state capture
    kubectl get hpa > "$METRICS_DIR/${TEST_NAME}_hpa_final.txt"
    kubectl top pods > "$METRICS_DIR/${TEST_NAME}_pods_final.txt" 2>/dev/null
}

# Function to run test pattern 1: Variable Load with Custom Shape
run_pattern_1() {
    echo "=== Running Pattern 1: Variable Load Shapes ==="
    
    cat > "/tmp/load_shape_${TEST_NAME}.py" << 'EOF'
from locust import HttpUser, task, between, LoadTestShape
import time
import math
import random

class CustomLoadShape(LoadTestShape):
    stages = [
        {"duration": 120, "users": 50, "spawn_rate": 2},   # Ramp up
        {"duration": 240, "users": 80, "spawn_rate": 3},   # Sustained
        {"duration": 360, "users": 20, "spawn_rate": 5},   # Drop
        {"duration": 480, "users": 100, "spawn_rate": 4},  # Spike
        {"duration": 600, "users": 40, "spawn_rate": 2},   # Stabilize
    ]
    
    def tick(self):
        run_time = self.get_run_time()
        
        for stage in self.stages:
            if run_time < stage["duration"]:
                # Add some randomness for more realistic load
                users = stage["users"] + random.randint(-10, 10)
                users = max(5, users)  # Minimum 5 users
                return (users, stage["spawn_rate"])
        
        return None

class WebUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def homepage(self):
        self.client.get("/")
    
    @task(2)
    def product_page(self):
        products = ["OLJCESPC7Z", "66VCHSJNUP", "1YMWWN1N4O"]
        self.client.get(f"/product/{random.choice(products)}")
    
    @task(1)
    def cart_operations(self):
        self.client.get("/cart")
        if random.random() < 0.3:
            self.client.post("/cart", data={"product_id": "OLJCESPC7Z", "quantity": 1})
EOF

    echo "Starting Locust with custom load shape..."
    locust -f "/tmp/load_shape_${TEST_NAME}.py" \
           --host="$FRONTEND_URL" \
           --headless \
           --run-time 600s \
           --html "$METRICS_DIR/${TEST_NAME}_pattern1_report.html" \
           --csv "$METRICS_DIR/${TEST_NAME}_pattern1" \
           --logfile "$METRICS_DIR/${TEST_NAME}_pattern1.log" &
    
    LOCUST_PID=$!
    echo "Locust started (PID: $LOCUST_PID)"
    
    # Wait for test to complete
    wait $LOCUST_PID
    echo "Pattern 1 completed"
}

# Function to run test pattern 2: Oscillating Load
run_pattern_2() {
    echo "=== Running Pattern 2: Oscillating Load ==="
    echo "This pattern will create sine wave load for 8 minutes"
    
    for i in {1..48}; do  # 48 iterations * 10 seconds = 8 minutes
        # Calculate sine wave (10-90 users)
        angle=$(echo "scale=2; $i * 0.26" | bc)  # 0.26 radians ≈ 15 degrees
        sine_val=$(echo "scale=2; s($angle)" | bc -l)
        users=$(echo "scale=0; 50 + $sine_val * 40" | bc)
        users=${users%.*}  # Remove decimal
        
        echo "Iteration $i/48: Starting $users users"
        
        # Start load test for 10 seconds
        timeout 10s locust -f "/home/mohamed/Desktop/Pfe_new/organized_new/advanced_variable_load.py" \
                          --host="$FRONTEND_URL" \
                          --headless \
                          --users "$users" \
                          --spawn-rate 5 \
                          --run-time 10s > /dev/null 2>&1 &
        
        sleep 10
        
        # Kill any remaining locust processes
        pkill -f "locust.*$FRONTEND_URL" 2>/dev/null || true
        
        echo "Current HPA status:"
        kubectl get hpa | tail -n +2
        echo "---"
    done
    
    echo "Pattern 2 completed"
}

# Function to run test pattern 3: Spike Pattern
run_pattern_3() {
    echo "=== Running Pattern 3: Random Spikes ==="
    echo "This pattern creates random load spikes for 10 minutes"
    
    for i in {1..30}; do  # 30 cycles of 20 seconds each
        # Random spike decision
        if [ $((RANDOM % 100)) -lt 40 ]; then  # 40% chance of spike
            users=$((80 + RANDOM % 41))  # 80-120 users
            duration=15
            echo "SPIKE: $users users for ${duration}s"
        else
            users=$((10 + RANDOM % 21))  # 10-30 users
            duration=20
            echo "Normal: $users users for ${duration}s"
        fi
        
        # Run the load
        timeout ${duration}s locust -f "/home/mohamed/Desktop/Pfe_new/organized_new/advanced_variable_load.py" \
                                   --host="$FRONTEND_URL" \
                                   --headless \
                                   --users "$users" \
                                   --spawn-rate 8 \
                                   --run-time ${duration}s > /dev/null 2>&1 &
        
        sleep $duration
        pkill -f "locust.*$FRONTEND_URL" 2>/dev/null || true
        
        echo "HPA Status after iteration $i:"
        kubectl get hpa --no-headers | awk '{print $1, $3, $4, $5, $6}'
        echo "---"
    done
    
    echo "Pattern 3 completed"
}

# Main execution
main() {
    echo "Starting advanced load test at $(date)"
    
    # Check prerequisites
    if ! command -v locust &> /dev/null; then
        echo "Error: Locust is not installed. Install with: pip install locust"
        exit 1
    fi
    
    if ! check_frontend; then
        exit 1
    fi
    
    # Start metrics collection
    start_metrics_collection
    
    # Trap to ensure cleanup
    trap 'stop_metrics_collection; pkill -f "locust.*$FRONTEND_URL" 2>/dev/null || true; exit' INT TERM EXIT
    
    echo "=== Test Configuration ==="
    echo "Total estimated time: ~20 minutes"
    echo "Pattern 1: Variable Load Shapes (10 min)"
    echo "Pattern 2: Oscillating Load (8 min)"
    echo "Pattern 3: Random Spikes (10 min)"
    echo "================================="
    
    read -p "Press Enter to start the test..."
    
    # Run all patterns
    run_pattern_1
    sleep 30  # Cooldown between patterns
    
    run_pattern_2
    sleep 30  # Cooldown between patterns
    
    run_pattern_3
    
    # Final cleanup
    stop_metrics_collection
    pkill -f "locust.*$FRONTEND_URL" 2>/dev/null || true
    
    echo "=== Test Completed ==="
    echo "Metrics saved in: $METRICS_DIR"
    echo "Test name: $TEST_NAME"
    echo "Files generated:"
    ls -la "$METRICS_DIR"/*${TEST_NAME}*
    
    echo ""
    echo "To analyze results:"
    echo "1. Check HPA events: kubectl describe hpa"
    echo "2. Review metrics: ls $METRICS_DIR/"
    echo "3. View scaling events: kubectl get events --sort-by=.metadata.creationTimestamp"
}

# Run main function
main "$@"
