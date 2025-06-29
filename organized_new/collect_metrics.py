#!/usr/bin/env python3
"""
Prometheus Metrics Collector for Online Boutique Services
Collects CPU, memory, pod count, desired replicas, and network traffic metrics for all services.
"""

import requests
import csv
import time
import argparse
from datetime import datetime
import sys

# Prometheus server URL (adjust if needed)
PROMETHEUS_URL = "http://localhost:9095"

# Kubernetes namespace for Online Boutique
NAMESPACE = "default"

# All Online Boutique services
SERVICES = [
    'frontend',
    'cartservice', 
    'productcatalogservice',
    'currencyservice',
    'paymentservice',
    'shippingservice',
    'emailservice',
    'checkoutservice',
    'recommendationservice',
    'adservice',
    'redis-cart'
]

def query_prometheus(query, timestamp=None):
    """
    Query Prometheus and return the result.
    Returns None if query fails or no data found.
    """
    try:
        url = f"{PROMETHEUS_URL}/api/v1/query"
        params = {'query': query}
        if timestamp:
            params['time'] = timestamp
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data['status'] == 'success' and data['data']['result']:
            return data['data']['result']
        return None
    except Exception:
        # Suppress errors for cleaner output
        return None

def get_cpu_usage(service_name):
    """Get CPU usage for specified service pods using irate."""
    # Get pod names for the service first
    pod_query = f'kube_pod_info{{namespace="{NAMESPACE}", pod=~"{service_name}-.*"}}'
    pod_result = query_prometheus(pod_query)
    
    if not pod_result:
        return 0.0
    
    total_cpu = 0.0
    pod_count = 0
    
    for pod_info in pod_result:
        pod_name = pod_info['metric']['pod']
        
        # Use irate query for CPU usage
        query_cpu = f'sum(irate(container_cpu_usage_seconds_total{{namespace="{NAMESPACE}", pod="{pod_name}"}}[5m])) by (pod)'
        result = query_prometheus(query_cpu)
        
        if result and result[0]['value'][1]:
            total_cpu += float(result[0]['value'][1]) * 100  # Convert to percentage
            pod_count += 1
    
    return round(total_cpu, 2) if pod_count > 0 else 0.0

def get_memory_usage(service_name):
    """Get memory usage for specified service pods using irate."""
    # Get pod names for the service first
    pod_query = f'kube_pod_info{{namespace="{NAMESPACE}", pod=~"{service_name}-.*"}}'
    pod_result = query_prometheus(pod_query)
    
    if not pod_result:
        return 0.0
    
    total_memory = 0.0
    pod_count = 0
    
    for pod_info in pod_result:
        pod_name = pod_info['metric']['pod']
        
        # Use irate query for memory usage
        query_mem = f'sum(irate(container_memory_working_set_bytes{{namespace="{NAMESPACE}", pod="{pod_name}"}}[5m])) by (pod)'
        result = query_prometheus(query_mem)
        
        if result and result[0]['value'][1]:
            total_memory += float(result[0]['value'][1]) / 1024 / 1024  # Convert to MB
            pod_count += 1
    
    return round(total_memory, 2) if pod_count > 0 else 0.0

def get_pod_count(service_name):
    """Get current number of pods for specified service."""
    query = f'count(kube_pod_info{{namespace="{NAMESPACE}", pod=~"{service_name}-.*"}})'
    result = query_prometheus(query)
    
    if result and result[0]['value'][1]:
        return int(float(result[0]['value'][1]))
    return 0

def get_desired_replicas(service_name):
    """Get desired replicas from HPA or Deployment for specified service."""
    # Try HPA first, then Deployment
    queries = [
        # HPA desired replicas
        f'kube_horizontalpodautoscaler_status_desired_replicas{{horizontalpodautoscaler=~"{service_name}.*"}}',
        # Deployment spec replicas
        f'kube_deployment_spec_replicas{{deployment=~"{service_name}.*"}}',
        # Fallback: ReplicaSet spec replicas
        f'kube_replicaset_spec_replicas{{replicaset=~"{service_name}.*"}}'
    ]
    
    for query in queries:
        result = query_prometheus(query)
        if result and result[0]['value'][1]:
            return int(float(result[0]['value'][1]))
    
    return 1  # Default to 1 if no HPA/Deployment found

def get_traffic_in(service_name):
    """Get network traffic in (bytes received) for specified service pods using irate."""
    # Get pod names for the service first
    pod_query = f'kube_pod_info{{namespace="{NAMESPACE}", pod=~"{service_name}-.*"}}'
    pod_result = query_prometheus(pod_query)
    
    if not pod_result:
        return 0.0
    
    total_traffic = 0.0
    pod_count = 0
    
    for pod_info in pod_result:
        pod_name = pod_info['metric']['pod']
        
        # Use irate query for network receive bytes
        query_received = f'sum(irate(container_network_receive_bytes_total{{namespace="{NAMESPACE}", pod="{pod_name}"}}[5m])) by (pod)'
        result = query_prometheus(query_received)
        
        if result and result[0]['value'][1]:
            total_traffic += float(result[0]['value'][1])
            pod_count += 1
    
    # If no container-level metrics found, use fallback
    if total_traffic == 0.0:
        query = 'sum(rate(node_network_receive_bytes_total{device!="lo"}[1m]))'
        result = query_prometheus(query)
        if result and result[0]['value'][1]:
            total_traffic = float(result[0]['value'][1]) * 0.1  # Approximate service share
    
    return round(total_traffic, 2)

def get_traffic_out(service_name):
    """Get network traffic out (bytes transmitted) for specified service pods using irate."""
    # Get pod names for the service first
    pod_query = f'kube_pod_info{{namespace="{NAMESPACE}", pod=~"{service_name}-.*"}}'
    pod_result = query_prometheus(pod_query)
    
    if not pod_result:
        return 0.0
    
    total_traffic = 0.0
    pod_count = 0
    
    for pod_info in pod_result:
        pod_name = pod_info['metric']['pod']
        
        # Use irate query for network transmit bytes
        query_transmit = f'sum(irate(container_network_transmit_bytes_total{{namespace="{NAMESPACE}", pod="{pod_name}"}}[5m])) by (pod)'
        result = query_prometheus(query_transmit)
        
        if result and result[0]['value'][1]:
            total_traffic += float(result[0]['value'][1])
            pod_count += 1
    
    # If no container-level metrics found, use fallback
    if total_traffic == 0.0:
        query = 'sum(rate(node_network_transmit_bytes_total{device!="lo"}[1m]))'
        result = query_prometheus(query)
        if result and result[0]['value'][1]:
            total_traffic = float(result[0]['value'][1]) * 0.1  # Approximate service share
    
    return round(total_traffic, 2)

def collect_all_metrics():
    """Collect all metrics for all services and return as a dictionary."""
    timestamp = datetime.now()
    
    # Base data with timestamp
    data = {
        'date': timestamp.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Collect metrics for each service
    for service in SERVICES:
        service_key = service.replace('-', '_')  # Replace hyphens with underscores for CSV columns
        
        data[f'{service_key}_num_pods'] = get_pod_count(service)
        data[f'{service_key}_desired_replicas'] = get_desired_replicas(service)
        data[f'{service_key}_cpu_usage'] = get_cpu_usage(service)
        data[f'{service_key}_mem_usage'] = get_memory_usage(service)
        data[f'{service_key}_traffic_in'] = get_traffic_in(service)
        data[f'{service_key}_traffic_out'] = get_traffic_out(service)
        # Add latency as 0 for now (placeholder for future implementation)
        data[f'{service_key}_latency'] = 0.0
    
    return data

def print_metrics(metrics):
    """Print metrics in a formatted way for all services."""
    print(f"\n[{metrics['date']}] ALL SERVICES METRICS:")
    print("=" * 80)
    
    for service in SERVICES:
        service_key = service.replace('-', '_')
        pods = metrics[f'{service_key}_num_pods']
        desired = metrics[f'{service_key}_desired_replicas']
        cpu = metrics[f'{service_key}_cpu_usage']
        memory = metrics[f'{service_key}_mem_usage']
        traffic_in = metrics[f'{service_key}_traffic_in']
        traffic_out = metrics[f'{service_key}_traffic_out']
        
        print(f"{service.upper():20} | Pods: {pods:2}/{desired:2} | "
              f"CPU: {cpu:6.2f}% | Mem: {memory:7.2f} MB | "
              f"In: {traffic_in:8.2f} B/s | Out: {traffic_out:8.2f} B/s")

def main():
    parser = argparse.ArgumentParser(description='Collect Prometheus metrics for all Online Boutique services')
    parser.add_argument('--interval', type=int, default=10, help='Collection interval in seconds (default: 10)')
    parser.add_argument('--duration', type=int, default=300, help='Total collection duration in seconds (default: 300)')
    parser.add_argument('--output', type=str, default=None, help='CSV output file (default: auto-generated)')
    parser.add_argument('--single', action='store_true', help='Collect metrics once and exit')
    
    args = parser.parse_args()
    
    # Generate default output filename if not provided
    if not args.output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'online_boutique_metrics_{timestamp}.csv'
    
    # Check Prometheus connectivity
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/label/__name__/values", timeout=5)
        response.raise_for_status()
        print(f"✓ Connected to Prometheus at {PROMETHEUS_URL}")
        print(f"✓ Monitoring {len(SERVICES)} services: {', '.join(SERVICES)}")
    except Exception as e:
        print(f"✗ Failed to connect to Prometheus: {e}")
        return 1
    
    collected_data = []
    
    if args.single:
        # Single collection
        print("Collecting metrics (single shot)...")
        metrics = collect_all_metrics()
        print_metrics(metrics)
        collected_data.append(metrics)
    else:
        # Continuous collection
        print(f"Starting metrics collection...")
        print(f"Interval: {args.interval}s, Duration: {args.duration}s, Output: {args.output}")
        print("Press Ctrl+C to stop early\n")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < args.duration:
                metrics = collect_all_metrics()
                print_metrics(metrics)
                collected_data.append(metrics)
                
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            print("\n\nCollection stopped by user")
    
    # Save to CSV
    if collected_data:
        print(f"\nSaving {len(collected_data)} records to {args.output}")
        
        # Create column headers for all services
        fieldnames = ['date']
        for service in SERVICES:
            service_key = service.replace('-', '_')
            fieldnames.extend([
                f'{service_key}_num_pods',
                f'{service_key}_desired_replicas', 
                f'{service_key}_cpu_usage',
                f'{service_key}_mem_usage',
                f'{service_key}_traffic_in',
                f'{service_key}_traffic_out',
                f'{service_key}_latency'
            ])
        
        with open(args.output, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for metrics in collected_data:
                writer.writerow(metrics)
        
        # Print summary statistics for a few key services
        if len(collected_data) > 1:
            print("\n--- Summary Statistics (Key Services) ---")
            
            key_services = ['frontend', 'cartservice', 'recommendationservice']
            for service in key_services:
                if service in SERVICES:
                    service_key = service.replace('-', '_')
                    pods_values = [m[f'{service_key}_num_pods'] for m in collected_data]
                    cpu_values = [m[f'{service_key}_cpu_usage'] for m in collected_data]
                    
                    print(f"{service.upper()}: "
                          f"Pods avg={sum(pods_values)/len(pods_values):.1f}, "
                          f"CPU avg={sum(cpu_values)/len(cpu_values):.2f}%")
        
        print(f"✓ Metrics saved to {args.output}")
    else:
        print("No data collected")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
