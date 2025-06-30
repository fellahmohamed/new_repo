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
    print(f"Testing connectivity to Istio Prometheus at {ISTIO_PROMETHEUS_URL}")
    print("=" * 60)
    
    try:
        # Test basic connectivity
        response = requests.get(f"{ISTIO_PROMETHEUS_URL}/api/v1/label/__name__/values", timeout=5)
        response.raise_for_status()
        
        data = response.json()
        if data['status'] == 'success':
            metrics = data['data']
            istio_metrics = [m for m in metrics if m.startswith('istio_')]
            
            print(f"✓ Connected successfully!")
            print(f"✓ Total metrics available: {len(metrics)}")
            print(f"✓ Istio metrics available: {len(istio_metrics)}")
            
            print(f"\nAvailable Istio metrics:")
            for metric in sorted(istio_metrics)[:10]:  # Show first 10
                print(f"  - {metric}")
            if len(istio_metrics) > 10:
                print(f"  ... and {len(istio_metrics) - 10} more")
            
            return True
        else:
            print(f"✗ API error: {data.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return False

def test_istio_metrics_availability():
    """Test availability of key Istio metrics"""
    print(f"\nTesting Istio metrics availability")
    print("=" * 60)
    
    # Key Istio metrics to test
    test_queries = [
        "istio_requests_total",
        "istio_request_duration_milliseconds_bucket",
        "istio_request_duration_seconds_bucket", 
        "istio_tcp_connections_opened_total",
        "istio_tcp_connections_closed_total",
        "istio_request_bytes",
        "istio_response_bytes"
    ]
    
    for query in test_queries:
        print(f"\nTesting: {query}")
        result = query_prometheus(query)
        if result:
            print(f"  ✓ Available ({len(result)} time series)")
        else:
            print(f"  ✗ Not available or no data")

def test_service_specific_metrics():
    """Test Istio metrics for specific services"""
    print(f"\nTesting service-specific Istio metrics")
    print("=" * 60)
    
    # Test a few key services
    test_services = ['frontend', 'productcatalogservice', 'cartservice']
    
    for service in test_services:
        print(f"\n--- Testing service: {service} ---")
        
        # Request rate
        query = f'sum(rate(istio_requests_total{{destination_service_name="{service}"}}[1m]))'
        print(f"\n1. Request rate:")
        result = query_prometheus(query)
        
        # Error rate  
        query = f'sum(rate(istio_requests_total{{destination_service_name="{service}", response_code!~"2.*"}}[1m])) / sum(rate(istio_requests_total{{destination_service_name="{service}"}}[1m])) * 100'
        print(f"\n2. Error rate:")
        result = query_prometheus(query)
        
        # P99 latency (try both metrics)
        print(f"\n3. P99 latency (milliseconds):")
        query = f'histogram_quantile(0.99, sum(rate(istio_request_duration_milliseconds_bucket{{destination_service_name="{service}"}}[1m])) by (le))'
        result = query_prometheus(query)
        
        if not result:
            print(f"\n3b. P99 latency (seconds * 1000):")
            query = f'histogram_quantile(0.99, sum(rate(istio_request_duration_seconds_bucket{{destination_service_name="{service}"}}[1m])) by (le)) * 1000'
            result = query_prometheus(query)

def test_label_exploration():
    """Explore available labels and their values"""
    print(f"\nExploring Istio metric labels")
    print("=" * 60)
    
    # Get labels for istio_requests_total
    print("\n1. Labels for istio_requests_total:")
    try:
        url = f"{ISTIO_PROMETHEUS_URL}/api/v1/label/__name__/values"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        # Try to get series info for istio_requests_total
        url = f"{ISTIO_PROMETHEUS_URL}/api/v1/series"
        params = {'match[]': 'istio_requests_total'}
        response = requests.get(url, params=params, timeout=5)
        series_data = response.json()
        
        if series_data['status'] == 'success' and series_data['data']:
            print(f"Found {len(series_data['data'])} time series")
            
            # Show unique label keys
            all_labels = set()
            for series in series_data['data'][:5]:  # Look at first 5 series
                all_labels.update(series.keys())
                print(f"  Sample: {series}")
            
            print(f"Available labels: {sorted(all_labels)}")
            
    except Exception as e:
        print(f"Error exploring labels: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test Istio metrics collection from Prometheus')
    # parser.add_argument('--url', default=ISTIO_PROMETHEUS_URL, help='Istio Prometheus URL')

    parser.add_argument('--service', help='Test specific service only')
    parser.add_argument('--verbose', action='store_true', help='Show detailed responses')
    
    args = parser.parse_args()
    
    # global ISTIO_PROMETHEUS_URL
    # ISTIO_PROMETHEUS_URL = args.url
    
    print(f"Istio Metrics Test")
    print(f"Target: {ISTIO_PROMETHEUS_URL}")
    print(f"Time: {datetime.now()}")
    print("=" * 60)
    
    # Test connectivity first
    if not test_connectivity():
        print("\n❌ Cannot proceed - connectivity test failed")
        return 1
    
    # Test metrics availability
    test_istio_metrics_availability()
    
    # Test service-specific metrics
    if args.service:
        print(f"\nTesting specific service: {args.service}")
        test_services = [args.service]
        # Run the service test with just one service
        for service in test_services:
            print(f"\n--- Testing service: {service} ---")
            
            queries = [
                (f'Request rate', f'sum(rate(istio_requests_total{{destination_service_name="{service}"}}[1m]))'),
                (f'Error rate', f'sum(rate(istio_requests_total{{destination_service_name="{service}", response_code!~"2.*"}}[1m])) / sum(rate(istio_requests_total{{destination_service_name="{service}"}}[1m])) * 100'),
                (f'P99 latency (ms)', f'histogram_quantile(0.99, sum(rate(istio_request_duration_milliseconds_bucket{{destination_service_name="{service}"}}[1m])) by (le))'),
                (f'P99 latency (s*1000)', f'histogram_quantile(0.99, sum(rate(istio_request_duration_seconds_bucket{{destination_service_name="{service}"}}[1m])) by (le)) * 1000')
            ]
            
            for name, query in queries:
                print(f"\n{name}:")
                query_prometheus(query, show_response=args.verbose)
    else:
        test_service_specific_metrics()
    
    # Explore labels
    test_label_exploration()
    
    print(f"\n🏁 Test completed!")
    return 0

if __name__ == "__main__":
    exit(main())
