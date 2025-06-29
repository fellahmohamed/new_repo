#!/bin/bash

# Usage: ./basic_metrics.sh <test_name> <duration_seconds>
TEST_NAME=${1:-"test_run"}
DURATION=${2:-300}
INTERVAL=10

echo "Starting basic metrics collection for: $TEST_NAME"
mkdir -p "metrics_data/$TEST_NAME"

# Enhanced CSV header for resource usage and idle percentage calculation
echo "timestamp,total_pods,cpu_used_millicores,memory_used_mb,cpu_allocatable_millicores,memory_allocatable_mb,cpu_usage_percent,memory_usage_percent,cpu_idle_percent,memory_idle_percent,cpu_requests_percent,memory_requests_percent" > "metrics_data/$TEST_NAME/basic_metrics.csv"

END_TIME=$(($(date +%s) + DURATION))

while [ $(date +%s) -lt $END_TIME ]; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Count total running pods
    TOTAL_PODS=$(kubectl get pods --no-headers | grep "Running" | wc -l)
    
    # Get actual pod resource usage (current consumption)
    TOTAL_CPU=$(kubectl top pods --no-headers --all-namespaces 2>/dev/null | awk '{sum += $3} END {gsub(/m/, "", sum); print sum+0}')
    TOTAL_MEMORY=$(kubectl top pods --no-headers --all-namespaces 2>/dev/null | awk '{sum += $4} END {gsub(/Mi/, "", sum); print sum+0}')
    
    # Get total allocatable resources across all nodes (fixed parsing)
    CPU_ALLOCATABLE=$(kubectl describe nodes 2>/dev/null | grep -A 10 "Allocatable:" | grep "cpu:" | awk '{
        if($2 ~ /m$/) {
            gsub(/m/, "", $2); 
            sum += $2
        } else {
            sum += $2 * 1000
        }
    } END {print sum+0}')
    
    MEMORY_ALLOCATABLE=$(kubectl describe nodes 2>/dev/null | grep -A 10 "Allocatable:" | grep "memory:" | awk '{
        if($2 ~ /Ki$/) {
            gsub(/Ki/, "", $2); 
            sum += $2/1024
        } else if($2 ~ /Mi$/) {
            gsub(/Mi/, "", $2); 
            sum += $2
        } else if($2 ~ /Gi$/) {
            gsub(/Gi/, "", $2); 
            sum += $2*1024
        }
    } END {print sum+0}')
    
    # Calculate usage percentages
    CPU_USAGE_PERCENT=$(echo "scale=2; if($CPU_ALLOCATABLE > 0) $TOTAL_CPU * 100 / $CPU_ALLOCATABLE else 0" | bc -l 2>/dev/null || echo "0")
    MEMORY_USAGE_PERCENT=$(echo "scale=2; if($MEMORY_ALLOCATABLE > 0) $TOTAL_MEMORY * 100 / $MEMORY_ALLOCATABLE else 0" | bc -l 2>/dev/null || echo "0")
    
    # Calculate idle percentages
    CPU_IDLE_PERCENT=$(echo "scale=2; 100 - $CPU_USAGE_PERCENT" | bc -l 2>/dev/null || echo "100")
    MEMORY_IDLE_PERCENT=$(echo "scale=2; 100 - $MEMORY_USAGE_PERCENT" | bc -l 2>/dev/null || echo "100")
    
    # Get cluster resource requests (what pods have requested)
    CLUSTER_CPU_REQUESTS=$(kubectl describe nodes 2>/dev/null | grep -A 2 "Allocated resources:" | grep "cpu" | awk '{print $2}' | sed 's/(//' | sed 's/%)//' | awk '{sum += $1; count++} END {if(count>0) print sum/count; else print 0}')
    CLUSTER_MEMORY_REQUESTS=$(kubectl describe nodes 2>/dev/null | grep -A 3 "Allocated resources:" | grep "memory" | awk '{print $2}' | sed 's/(//' | sed 's/%)//' | awk '{sum += $1; count++} END {if(count>0) print sum/count; else print 0}')
    
    echo "$TIMESTAMP,$TOTAL_PODS,$TOTAL_CPU,$TOTAL_MEMORY,$CPU_ALLOCATABLE,$MEMORY_ALLOCATABLE,$CPU_USAGE_PERCENT,$MEMORY_USAGE_PERCENT,$CPU_IDLE_PERCENT,$MEMORY_IDLE_PERCENT,$CLUSTER_CPU_REQUESTS,$CLUSTER_MEMORY_REQUESTS" >> "metrics_data/$TEST_NAME/basic_metrics.csv"
    
    echo "[$TIMESTAMP] Pods: $TOTAL_PODS | CPU: ${TOTAL_CPU}m/${CPU_ALLOCATABLE}m (${CPU_USAGE_PERCENT}% used, ${CPU_IDLE_PERCENT}% idle) | Memory: ${TOTAL_MEMORY}Mi/${MEMORY_ALLOCATABLE}Mi (${MEMORY_USAGE_PERCENT}% used, ${MEMORY_IDLE_PERCENT}% idle)"
    
    sleep $INTERVAL
done

echo "Basic metrics collection completed for $TEST_NAME"

# Calculate totals for the run
echo "=== RUN SUMMARY ===" >> "metrics_data/$TEST_NAME/summary.txt"
echo "Test: $TEST_NAME" >> "metrics_data/$TEST_NAME/summary.txt"
echo "Duration: $DURATION seconds" >> "metrics_data/$TEST_NAME/summary.txt"
echo "Collection completed at: $(date)" >> "metrics_data/$TEST_NAME/summary.txt"
echo "" >> "metrics_data/$TEST_NAME/summary.txt"

# Calculate average resource usage and idle percentages
echo "=== RESOURCE UTILIZATION SUMMARY ===" >> "metrics_data/$TEST_NAME/summary.txt"
AVG_CPU_USAGE=$(tail -n +2 "metrics_data/$TEST_NAME/basic_metrics.csv" | awk -F',' '{sum+=$7; count++} END {if(count>0) printf "%.2f", sum/count; else print "0"}')
AVG_MEMORY_USAGE=$(tail -n +2 "metrics_data/$TEST_NAME/basic_metrics.csv" | awk -F',' '{sum+=$8; count++} END {if(count>0) printf "%.2f", sum/count; else print "0"}')
AVG_CPU_IDLE=$(tail -n +2 "metrics_data/$TEST_NAME/basic_metrics.csv" | awk -F',' '{sum+=$9; count++} END {if(count>0) printf "%.2f", sum/count; else print "100"}')
AVG_MEMORY_IDLE=$(tail -n +2 "metrics_data/$TEST_NAME/basic_metrics.csv" | awk -F',' '{sum+=$10; count++} END {if(count>0) printf "%.2f", sum/count; else print "100"}')

echo "Average CPU Usage: ${AVG_CPU_USAGE}%" >> "metrics_data/$TEST_NAME/summary.txt"
echo "Average Memory Usage: ${AVG_MEMORY_USAGE}%" >> "metrics_data/$TEST_NAME/summary.txt"
echo "Average CPU Idle: ${AVG_CPU_IDLE}%" >> "metrics_data/$TEST_NAME/summary.txt"
echo "Average Memory Idle: ${AVG_MEMORY_IDLE}%" >> "metrics_data/$TEST_NAME/summary.txt"

echo ""
echo "=== Test Summary ==="
echo "Average CPU Usage: ${AVG_CPU_USAGE}% (Idle: ${AVG_CPU_IDLE}%)"
echo "Average Memory Usage: ${AVG_MEMORY_USAGE}% (Idle: ${AVG_MEMORY_IDLE}%)"
echo "Results saved to: metrics_data/$TEST_NAME/"