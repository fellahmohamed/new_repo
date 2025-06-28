#!/bin/bash

# Quick Aggressive Load Test Script
# Usage: ./quick_aggressive_test.sh [host_url]

HOST_URL=${1:-"http://192.168.49.2:31858"}
TEST_NAME="aggressive_test_$(date +%H%M%S)"

echo "🚀 Starting 3-minute aggressive load test..."
echo "Host: $HOST_URL"
echo "Test: $TEST_NAME"
echo ""

# Apply aggressive HPA configuration
echo "📋 Applying aggressive HPA configuration..."
kubectl apply -f hpa-config.yaml

echo "⏳ Waiting 10 seconds for HPA to initialize..."
sleep 10

# Start metrics collection in background
echo "📊 Starting metrics collection..."
./basic_metrics.sh "$TEST_NAME" 180 &
METRICS_PID=$!

# Start HPA metrics collection
./collect_metrics.sh "$TEST_NAME" 180 &
HPA_METRICS_PID=$!

sleep 2

# Show initial pod state
echo "📈 Initial pod state:"
kubectl get pods | grep -E "(frontend|recommendation|cart|product)"
echo ""

# Start the aggressive load test
echo "🔥 Launching aggressive load test for 3 minutes..."
echo "Expected behavior:"
echo "  • 0-30s: Warmup (moderate load)"
echo "  • 30-60s: Spike 1 (high load, expect scale up)"
echo "  • 60-90s: Cooldown 1 (reduced load, expect scale down)"
echo "  • 90-120s: Spike 2 (high load, expect scale up again)"
echo "  • 120-150s: Cooldown 2 (reduced load)"
echo "  • 150-180s: Finale spike (maximum load)"
echo ""

# Monitor HPA in background
watch -n 2 'echo "=== HPA Status ==="; kubectl get hpa; echo ""; echo "=== Pod Counts ==="; kubectl get pods | grep -E "(frontend|recommendation|cart|product)" | grep Running | wc -l; echo "Total running pods: $(kubectl get pods | grep Running | wc -l)"' &
WATCH_PID=$!

# Run the load test
locust -f enhanced_locust.py \
    --host="$HOST_URL" \
    --users 200 \
    --spawn-rate 20 \
    --run-time 3m \
    --headless \
    --csv="metrics_data/$TEST_NAME/locust" \
    --html="metrics_data/$TEST_NAME/locust_report.html"

# Stop monitoring
kill $WATCH_PID 2>/dev/null

# Wait for metrics collection to complete
echo ""
echo "⏳ Waiting for metrics collection to complete..."
wait $METRICS_PID
wait $HPA_METRICS_PID

echo ""
echo "✅ Test completed!"
echo "📊 Final pod state:"
kubectl get pods | grep -E "(frontend|recommendation|cart|product)"

echo ""
echo "📈 HPA Status:"
kubectl get hpa

echo ""
echo "📁 Results saved in: metrics_data/$TEST_NAME/"
echo "📋 View HTML report: metrics_data/$TEST_NAME/locust_report.html"
