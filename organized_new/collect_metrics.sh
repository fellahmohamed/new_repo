#!/bin/bash

# Usage: ./collect_metrics.sh <test_name> <duration_seconds>
# Example: ./collect_metrics.sh "hpa_test_1" 300

TEST_NAME=${1:-"test_run"}
DURATION=${2:-300}  # Default 5 minutes
INTERVAL=10         # Collect metrics every 10 seconds

echo "Starting metrics collection for: $TEST_NAME"
echo "Duration: $DURATION seconds, Interval: $INTERVAL seconds"

# Create output directory
mkdir -p metrics_data/$TEST_NAME

# CSV headers
echo "timestamp,frontend_pods,recommendation_pods,frontend_cpu_target,frontend_memory_target,recommendation_cpu_target" > metrics_data/$TEST_NAME/hpa_metrics.csv
echo "timestamp,pod_name,cpu_millicores,memory_mb" > metrics_data/$TEST_NAME/pod_metrics.csv

# Start collection
END_TIME=$(($(date +%s) + DURATION))

while [ $(date +%s) -lt $END_TIME ]; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    
    # Get HPA metrics
    FRONTEND_PODS=$(kubectl get deployment frontend -o jsonpath='{.status.replicas}')
    RECOMMENDATION_PODS=$(kubectl get deployment recommendationservice -o jsonpath='{.status.replicas}')
    
    # Get HPA targets (will be empty if HPA not active)
    FRONTEND_CPU=$(kubectl get hpa frontend-hpa -o jsonpath='{.status.currentCPUUtilizationPercentage}' 2>/dev/null || echo "0")
    FRONTEND_MEM=$(kubectl get hpa frontend-hpa -o jsonpath='{.status.currentMemoryUtilizationPercentage}' 2>/dev/null || echo "0")
    RECOMMENDATION_CPU=$(kubectl get hpa recommendationservice-hpa -o jsonpath='{.status.currentCPUUtilizationPercentage}' 2>/dev/null || echo "0")
    
    echo "$TIMESTAMP,$FRONTEND_PODS,$RECOMMENDATION_PODS,$FRONTEND_CPU,$FRONTEND_MEM,$RECOMMENDATION_CPU" >> metrics_data/$TEST_NAME/hpa_metrics.csv
    
    echo "[$TIMESTAMP] Frontend: $FRONTEND_PODS pods, CPU: $FRONTEND_CPU% | Recommendation: $RECOMMENDATION_PODS pods, CPU: $RECOMMENDATION_CPU%"
    
    sleep $INTERVAL
done

echo "Metrics collection completed for $TEST_NAME"