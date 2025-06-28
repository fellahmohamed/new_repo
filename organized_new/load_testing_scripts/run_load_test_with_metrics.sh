#!/bin/bash

# Load Test with Automatic Metrics Collection
# Usage: ./run_load_test_with_metrics.sh [test_name] [users] [spawn_rate] [duration]

# Default parameters
TEST_NAME=${1:-"load_test_$(date +%Y%m%d_%H%M%S)"}
USERS=${2:-100}
SPAWN_RATE=${3:-2}
DURATION=${4:-"30s"}
HOST_URL=${5:-"http://192.168.49.2:31858"}

echo "========================================="
echo "🚀 Starting Load Test with Metrics Collection"
echo "========================================="
echo "Test Name: $TEST_NAME"
echo "Users: $USERS"
echo "Spawn Rate: $SPAWN_RATE"
echo "Duration: $DURATION"
echo "Host: $HOST_URL"
echo "========================================="

# Convert duration from Locust format (5m) to seconds for metrics script
DURATION_SECONDS=$(echo "$DURATION" | sed 's/m/*60/' | sed 's/s//' | bc 2>/dev/null || echo "300")

# Create test directory
mkdir -p "metrics_data/$TEST_NAME"

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Stopping metrics collection..."
    kill $METRICS_PID 2>/dev/null
    kill $BASIC_METRICS_PID 2>/dev/null
    wait $METRICS_PID 2>/dev/null
    wait $BASIC_METRICS_PID 2>/dev/null
    echo "✅ Cleanup completed"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Start basic metrics collection in background
echo "📊 Starting basic metrics collection..."
./basic_metrics.sh "$TEST_NAME" $DURATION_SECONDS &
BASIC_METRICS_PID=$!

# Start HPA-specific metrics collection in background
echo "📈 Starting HPA metrics collection..."
./collect_metrics.sh "$TEST_NAME" $DURATION_SECONDS &
METRICS_PID=$!

# Wait a moment for metrics collection to initialize
sleep 2

# Start the load test
echo "🔥 Starting Locust load test..."
echo "Command: locust -f enhanced_locust.py --host=$HOST_URL --users $USERS --spawn-rate $SPAWN_RATE --run-time $DURATION --headless"

locust -f enhanced_locust.py \
    --host="$HOST_URL" \
    --users $USERS \
    --spawn-rate $SPAWN_RATE \
    --run-time $DURATION \
    --headless \
    --csv="metrics_data/$TEST_NAME/locust" \
    --html="metrics_data/$TEST_NAME/locust_report.html"

LOCUST_EXIT_CODE=$?

# Wait for metrics collection to complete
echo "⏳ Waiting for metrics collection to complete..."
wait $BASIC_METRICS_PID
wait $METRICS_PID

echo ""
echo "========================================="
echo "✅ Test Completed: $TEST_NAME"
echo "========================================="

# Generate summary report
echo "📋 Generating summary report..."
cat > "metrics_data/$TEST_NAME/test_summary.txt" << EOF
=== LOAD TEST SUMMARY ===
Test Name: $TEST_NAME
Date: $(date)
Duration: $DURATION ($DURATION_SECONDS seconds)
Users: $USERS
Spawn Rate: $SPAWN_RATE
Host: $HOST_URL
Locust Exit Code: $LOCUST_EXIT_CODE

=== FILES GENERATED ===
- basic_metrics.csv: Resource usage and idle percentages
- hpa_metrics.csv: HPA scaling metrics
- pod_metrics.csv: Individual pod metrics
- locust_stats.csv: Locust request statistics
- locust_failures.csv: Locust failure statistics
- locust_report.html: Locust HTML report
- summary.txt: Basic metrics summary
- test_summary.txt: This file

=== QUICK ANALYSIS ===
EOF

# Add quick analysis from basic metrics if available
if [ -f "metrics_data/$TEST_NAME/basic_metrics.csv" ]; then
    echo "Resource Utilization:" >> "metrics_data/$TEST_NAME/test_summary.txt"
    tail -n +2 "metrics_data/$TEST_NAME/basic_metrics.csv" | awk -F',' '
    BEGIN {
        max_cpu_usage = 0; min_cpu_usage = 100; cpu_sum = 0; cpu_count = 0;
        max_mem_usage = 0; min_mem_usage = 100; mem_sum = 0; mem_count = 0;
    }
    {
        if ($7 != "" && $7 != "cpu_usage_percent") {
            cpu_usage = $7;
            if (cpu_usage > max_cpu_usage) max_cpu_usage = cpu_usage;
            if (cpu_usage < min_cpu_usage) min_cpu_usage = cpu_usage;
            cpu_sum += cpu_usage; cpu_count++;
        }
        if ($8 != "" && $8 != "memory_usage_percent") {
            mem_usage = $8;
            if (mem_usage > max_mem_usage) max_mem_usage = mem_usage;
            if (mem_usage < min_mem_usage) min_mem_usage = mem_usage;
            mem_sum += mem_usage; mem_count++;
        }
    }
    END {
        if (cpu_count > 0) {
            printf "CPU Usage - Min: %.2f%%, Max: %.2f%%, Avg: %.2f%%\n", min_cpu_usage, max_cpu_usage, cpu_sum/cpu_count;
        }
        if (mem_count > 0) {
            printf "Memory Usage - Min: %.2f%%, Max: %.2f%%, Avg: %.2f%%\n", min_mem_usage, max_mem_usage, mem_sum/mem_count;
        }
    }' >> "metrics_data/$TEST_NAME/test_summary.txt"
fi

echo ""
echo "📁 Results saved in: metrics_data/$TEST_NAME/"
echo "📊 View HTML report: metrics_data/$TEST_NAME/locust_report.html"
echo "📈 View test summary: metrics_data/$TEST_NAME/test_summary.txt"

# Show quick preview of results
if [ -f "metrics_data/$TEST_NAME/test_summary.txt" ]; then
    echo ""
    echo "=== QUICK PREVIEW ==="
    tail -n 5 "metrics_data/$TEST_NAME/test_summary.txt"
fi

echo "========================================="
