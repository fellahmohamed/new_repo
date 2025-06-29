#!/bin/bash

# Updated Quick Test with Port Forwarding
# This version uses port forwarding to ensure connectivity

echo "=== Quick Aggressive HPA Test (Port Forward Version) ==="

# Set up port forwarding
echo "Setting up port forwarding..."
kubectl port-forward svc/frontend-external 8080:80 &
PORT_FORWARD_PID=$!

# Wait for port forward to be ready
sleep 5

# Test connectivity
FRONTEND_URL="http://localhost:8080"
echo "Testing frontend at: $FRONTEND_URL"

if curl -s --max-time 5 "$FRONTEND_URL" > /dev/null; then
    echo "✓ Frontend is accessible via port forwarding"
else
    echo "✗ Frontend still not accessible"
    kill $PORT_FORWARD_PID 2>/dev/null
    exit 1
fi

# Create metrics directory
METRICS_DIR="/home/mohamed/Desktop/Pfe_new/organized_new/metrics_data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$METRICS_DIR"

# Apply aggressive HPA if not already applied
echo "Applying aggressive HPA..."
kubectl apply -f /home/mohamed/Desktop/Pfe_new/organized_new/aggressive-hpa.yaml

# Wait for HPA to initialize
sleep 10

echo "Current HPA status:"
kubectl get hpa

# Start monitoring
{
    while true; do
        echo "$(date): $(kubectl get hpa --no-headers | awk '{print $1, $3, $4, $5}')" >> "$METRICS_DIR/quick_test_portforward_${TIMESTAMP}.log"
        sleep 5
    done
} &
MONITOR_PID=$!

# Cleanup function
cleanup() {
    kill $MONITOR_PID 2>/dev/null
    kill $PORT_FORWARD_PID 2>/dev/null
    pkill -f "locust" 2>/dev/null || true
    echo "Test completed. Check $METRICS_DIR/quick_test_portforward_${TIMESTAMP}.log"
}
trap cleanup EXIT INT TERM

echo "Starting 5-minute aggressive load test..."

# Phase 1: Spike to 60 users (30 seconds)
echo "Phase 1: Spike to 60 users"
timeout 30s locust -f /home/mohamed/Desktop/Pfe_new/organized_new/enhanced_locust.py \
                   --host="$FRONTEND_URL" \
                   --headless \
                   --users 60 \
                   --spawn-rate 10 \
                   --run-time 30s > /dev/null 2>&1 &
sleep 35

# Phase 2: Drop to 5 users (30 seconds)
echo "Phase 2: Drop to 5 users"
timeout 30s locust -f /home/mohamed/Desktop/Pfe_new/organized_new/enhanced_locust.py \
                   --host="$FRONTEND_URL" \
                   --headless \
                   --users 5 \
                   --spawn-rate 5 \
                   --run-time 30s > /dev/null 2>&1 &
sleep 35

# Phase 3: Medium load (30 seconds)
echo "Phase 3: Medium load 30 users"
timeout 30s locust -f /home/mohamed/Desktop/Pfe_new/organized_new/enhanced_locust.py \
                   --host="$FRONTEND_URL" \
                   --headless \
                   --users 30 \
                   --spawn-rate 8 \
                   --run-time 30s > /dev/null 2>&1 &
sleep 35

# Phase 4: Final spike (30 seconds)
echo "Phase 4: Final spike to 80 users"
timeout 30s locust -f /home/mohamed/Desktop/Pfe_new/organized_new/enhanced_locust.py \
                   --host="$FRONTEND_URL" \
                   --headless \
                   --users 80 \
                   --spawn-rate 15 \
                   --run-time 30s > /dev/null 2>&1 &
sleep 35

# Phase 5: Gradual decrease (60 seconds)
echo "Phase 5: Gradual decrease"
for users in 40 20 10 5; do
    echo "  Setting $users users"
    timeout 15s locust -f /home/mohamed/Desktop/Pfe_new/organized_new/enhanced_locust.py \
                       --host="$FRONTEND_URL" \
                       --headless \
                       --users $users \
                       --spawn-rate 5 \
                       --run-time 15s > /dev/null 2>&1 &
    sleep 17
done

echo "=== Test Summary ==="
echo "Final HPA status:"
kubectl get hpa
echo ""
echo "Recent scaling events:"
kubectl get events --field-selector type=Normal | grep -E "(Scaled|HorizontalPodAutoscaler)" | tail -10
echo ""
echo "Log file: $METRICS_DIR/quick_test_portforward_${TIMESTAMP}.log"
