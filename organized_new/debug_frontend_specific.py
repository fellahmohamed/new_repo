#!/usr/bin/env python3
"""
Debug script to verify that metrics are specifically for frontend pods
"""

import requests

PROMETHEUS_URL = "http://localhost:9091"

def test_query(query, description):
    """Test a Prometheus query and show results."""
    print(f"\n=== {description} ===")
    print(f"Query: {query}")
    
    try:
        url = f"{PROMETHEUS_URL}/api/v1/query"
        params = {'query': query}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data['status'] == 'success' and data['data']['result']:
            print("✓ Query successful")
            for i, result in enumerate(data['data']['result']):
                if 'metric' in result:
                    pod_name = result['metric'].get('pod', 'N/A')
                    container = result['metric'].get('container', 'N/A')
                    value = result['value'][1]
                    print(f"  Result {i+1}: Pod={pod_name}, Container={container}, Value={value}")
                else:
                    print(f"  Result {i+1}: Value={result['value'][1]}")
            return True
        else:
            print("✗ No results found")
            return False
    except Exception as e:
        print(f"✗ Query failed: {e}")
        return False

# Test all our queries
queries = [
    # CPU queries
    ('sum(rate(container_cpu_usage_seconds_total{pod=~"frontend-.*", container!="POD", container!=""}[1m])) * 100', 'CPU Usage - Primary'),
    ('rate(container_cpu_usage_seconds_total{pod=~"frontend-.*", container!="POD", container!=""}[1m]) * 100', 'CPU Usage - Individual containers'),
    
    # Memory queries  
    ('sum(container_memory_working_set_bytes{pod=~"frontend-.*", container!="POD", container!=""}) / 1024 / 1024', 'Memory Usage - Primary'),
    ('container_memory_working_set_bytes{pod=~"frontend-.*", container!="POD", container!=""} / 1024 / 1024', 'Memory Usage - Individual containers'),
    
    # Pod count
    ('count(kube_pod_info{pod=~"frontend-.*"})', 'Pod Count'),
    ('kube_pod_info{pod=~"frontend-.*"}', 'Pod Info - Individual pods'),
    
    # Network traffic
    ('sum(rate(container_network_receive_bytes_total{pod=~"frontend-.*"}[1m]))', 'Network In - Container level'),
    ('rate(container_network_receive_bytes_total{pod=~"frontend-.*"}[1m])', 'Network In - Individual containers'),
    ('sum(rate(container_network_transmit_bytes_total{pod=~"frontend-.*"}[1m]))', 'Network Out - Container level'),
    ('rate(container_network_transmit_bytes_total{pod=~"frontend-.*"}[1m])', 'Network Out - Individual containers'),
]

print("Testing Prometheus queries for frontend-specific metrics...")
print("Current frontend pod: frontend-754cdbf884-vvc42")

for query, description in queries:
    test_query(query, description)

print("\n" + "="*60)
print("SUMMARY: Check that all metrics above show 'frontend' in pod names")
print("="*60)
