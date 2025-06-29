#!/bin/bash

# Quick Aggressive HPA Test
# Simplified version for quick testing and verification

FRONTEND_URL="http://$(minikube ip):$(kubectl get service frontend-external -o jsonpath='{.spec.ports[0].nodePort}')"
METRICS_DIR="/home/mohamed/Desktop/Pfe_new/organized_new/metrics_data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Quick Aggressive HPA Test ==="
echo "Frontend URL: $FRONTEND_URL"
echo "This test runs for 5 minutes with rapid load changes"

# Ensure metrics directory exists
mkdir -p "$METRICS_DIR"

# Check if frontend is accessible
echo "Checking frontend..."
if ! curl -s --max-time 5 "$FRONTEND_URL" > /dev/null; then
    echo "Error: Cannot reach frontend at $FRONTEND_URL"
    echo "Check: kubectl get svc frontend-external"
    exit 1
fi
echo "✓ Frontend accessible"

# Check HPA status
echo "Current HPA status:"
kubectl get hpa

# Start monitoring in background
{
    while true; do
        echo "$(date): $(kubectl get hpa --no-headers | awk '{print $1, $3, $4, $5}')" >> "$METRICS_DIR/quick_test_${TIMESTAMP}.log"
        sleep 5
    done
} &
MONITOR_PID=$!

# Cleanup function
cleanup() {
    kill $MONITOR_PID 2>/dev/null
    pkill -f "locust" 2>/dev/null || true
    echo "Test completed. Check $METRICS_DIR/quick_test_${TIMESTAMP}.log"
}
trap cleanup EXIT INT TERM

echo "Starting load test phases..."

# Phase 1: Sudden spike (30 seconds)
echo "Phase 1: Spike to 80 users"
timeout 30s locust -f /home/mohamed/Desktop/Pfe_new/organized_new/enhanced_locust.py \
                   --host="$FRONTEND_URL" \
                   --headless \
                   --users 80 \
                   --spawn-rate 10 \
                   --run-time 30s > /dev/null 2>&1 &
sleep 35

# Phase 2: Drop to baseline (30 seconds)
echo "Phase 2: Drop to 5 users"
timeout 30s locust -f /home/mohamed/Desktop/Pfe_new/organized_new/enhanced_locust.py \
                   --host="$FRONTEND_URL" \
                   --headless \
                   --users 5 \
                   --spawn-rate 5 \
                   --run-time 30s > /dev/null 2>&1 &
sleep 35

# Phase 3: Medium load (45 seconds)
echo "Phase 3: Medium load 40 users"
timeout 45s locust -f /home/mohamed/Desktop/Pfe_new/organized_new/enhanced_locust.py \
                   --host="$FRONTEND_URL" \
                   --headless \
                   --users 40 \
                   --spawn-rate 8 \
                   --run-time 45s > /dev/null 2>&1 &
sleep 50

# Phase 4: Another spike (30 seconds)
echo "Phase 4: Second spike to 100 users"
timeout 30s locust -f /home/mohamed/Desktop/Pfe_new/organized_new/enhanced_locust.py \
                   --host="$FRONTEND_URL" \
                   --headless \
                   --users 100 \
                   --spawn-rate 15 \
                   --run-time 30s > /dev/null 2>&1 &
sleep 35

# Phase 5: Gradual decrease (60 seconds)
echo "Phase 5: Gradual decrease"
for users in 60 40 20 10; do
    echo "  Setting $users users"
    timeout 15s locust -f /home/mohamed/Desktop/Pfe_new/organized_new/enhanced_locust.py \
                       --host="$FRONTEND_URL" \
                       --headless \
                       --users $users \
                       --spawn-rate 5 \
                       --run-time 15s > /dev/null 2>&1 &
    sleep 17
done

echo "=== Quick Test Summary ==="
echo "Final HPA status:"
kubectl get hpa
echo ""
echo "Recent scaling events:"
kubectl get events --field-selector type=Normal | grep -E "(Scaled|HorizontalPodAutoscaler)" | tail -10
echo ""
echo "Log file: $METRICS_DIR/quick_test_${TIMESTAMP}.log"

echo ""
echo "📈 HPA Status:"
kubectl get hpa

echo ""
echo "📁 Results saved in: metrics_data/$TEST_NAME/"
echo "📋 View HTML report: metrics_data/$TEST_NAME/locust_report.html"
