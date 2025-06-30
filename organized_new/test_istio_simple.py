#!/usr/bin/env python3
"""
Simple test script for Istio traffic metrics (traffic in/out) for one service
"""

import requests
import json

# Istio Prometheus URL
ISTIO_PROMETHEUS_URL = "http://localhost:9092"

# Test service
SERVICE_NAME = "frontend"

def query_prometheus(query):
    """Query Prometheus and return the result."""
    try:
        url = f"{ISTIO_PROMETHEUS_URL}/api/v1/query"
        params = {'query': query}
        
        print(f"Query: {query}")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] == 'success':
            result_count = len(data['data']['result'])
            print(f"✓ Success - {result_count} results")
            
            if data['data']['result']:
                for result in data['data']['result']:
                    value = result.get('value', [None, None])[1]
                    metric = result.get('metric', {})
                    print(f"  Value: {value}")
                    print(f"  Labels: {json.dumps(metric, indent=4)}")
                return data['data']['result']
            else:
                print("  No data returned")
                return []
        else:
            print(f"✗ Error: {data.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def test_connectivity():
    """Test basic connectivity to Istio Prometheus"""
    print(f"Testing connectivity to {ISTIO_PROMETHEUS_URL}")
    try:
        response = requests.get(f"{ISTIO_PROMETHEUS_URL}/api/v1/label/__name__/values", timeout=5)
        response.raise_for_status()
        print("✓ Connection successful")
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

def test_traffic_metrics():
    """Test traffic in and traffic out metrics for the service"""
    print(f"\nTesting traffic metrics for service: {SERVICE_NAME}")
    print("=" * 50)
    
    # Traffic IN - we found this works
    print("\n1. Testing Traffic IN metrics:")
    traffic_in_query = f'sum(rate(istio_requests_total{{destination_service_name="{SERVICE_NAME}"}}[1m]))'
    print(f"\n  Working query:")
    result = query_prometheus(traffic_in_query)
    if result and result[0]['value'][1]:
        traffic_in_value = float(result[0]['value'][1])
        print(f"  ✓ Traffic IN: {traffic_in_value:.2f} bytes/sec")
    
    # Traffic OUT - let's try response bytes with destination instead of source
    print("\n2. Testing Traffic OUT metrics:")
    traffic_out_queries = [
        f'sum(rate(istio_response_bytes_sum{{destination_service_name="{SERVICE_NAME}"}}[1m]))',
        f'sum(rate(istio_response_bytes_sum{{source_service_name="{SERVICE_NAME}"}}[1m]))',
        f'sum(rate(istio_response_bytes{{destination_service_name="{SERVICE_NAME}"}}[1m]))',
        f'sum(rate(istio_response_bytes{{source_service_name="{SERVICE_NAME}"}}[1m]))',
    ]
    
    for i, query in enumerate(traffic_out_queries):
        print(f"\n  Query variant {i+1}:")
        result = query_prometheus(query)
        if result and result[0]['value'][1] and float(result[0]['value'][1]) > 0:
            traffic_out_value = float(result[0]['value'][1])
            print(f"  ✓ Found non-zero traffic OUT: {traffic_out_value:.2f} bytes/sec")
            break
    
    # Let's also explore what labels are available for these metrics
    print("\n3. Exploring available metric labels:")
    
    # Get all istio_request_bytes_sum series
    print(f"\n  Labels for istio_request_bytes_sum:")
    explore_query = 'istio_request_bytes_sum'
    result = query_prometheus(explore_query)
    
    if result:
        print(f"  Found {len(result)} time series")
        for i, series in enumerate(result[:3]):  # Show first 3
            labels = series.get('metric', {})
            value = series.get('value', [None, None])[1]
            print(f"    [{i+1}] Value: {value}")
            print(f"        Labels: {json.dumps(labels, indent=8)}")
    
    # Get all istio_response_bytes_sum series  
    print(f"\n  Labels for istio_response_bytes_sum:")
    explore_query = 'istio_response_bytes_sum'
    result = query_prometheus(explore_query)
    
    if result:
        print(f"  Found {len(result)} time series")
        for i, series in enumerate(result[:3]):  # Show first 3
            labels = series.get('metric', {})
            value = series.get('value', [None, None])[1]
            print(f"    [{i+1}] Value: {value}")
            print(f"        Labels: {json.dumps(labels, indent=8)}")

def main():
    print("Simple Istio Traffic Metrics Test")
    print("=" * 50)
    
    # Test connectivity first
    if not test_connectivity():
        print("Cannot proceed - connection failed")
        return 1
    
    # Test traffic metrics
    test_traffic_metrics()
    
    print(f"\nTest completed!")
    return 0

if __name__ == "__main__":
    exit(main())
