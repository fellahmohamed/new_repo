#!/usr/bin/env python3
"""
Debug script to find network metrics
"""

import requests

def check_network_metrics():
    prometheus_url = "http://localhost:9091"
    query_url = f"{prometheus_url}/api/v1/query"
    
    print("🔍 Checking for network metrics...")
    
    # Test different network metric patterns
    queries = [
        '{__name__=~".*network.*"}',
        '{__name__=~"container_network.*"}', 
        '{__name__=~".*receive.*"}',
        '{__name__=~".*transmit.*"}',
        '{__name__=~".*bytes.*"}',
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        try:
            params = {'query': query}
            response = requests.get(query_url, params=params)
            data = response.json()
            
            if data['status'] == 'success' and data['data']['result']:
                print(f"Found {len(data['data']['result'])} metrics:")
                for result in data['data']['result'][:5]:  # Show first 5
                    metric_name = result['metric']['__name__']
                    labels = {k: v for k, v in result['metric'].items() if k != '__name__'}
                    print(f"  - {metric_name} {labels}")
                if len(data['data']['result']) > 5:
                    print(f"  ... and {len(data['data']['result']) - 5} more")
            else:
                print("No results found")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    check_network_metrics()
