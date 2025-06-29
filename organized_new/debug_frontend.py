#!/usr/bin/env python3
"""
Debug script to find frontend-specific metrics
"""

import requests

def check_frontend_metrics():
    prometheus_url = "http://localhost:9091"
    query_url = f"{prometheus_url}/api/v1/query"
    
    print("🔍 Checking for frontend-specific metrics...")
    
    # Test different patterns for frontend
    queries = [
        '{pod=~"frontend.*"}',
        '{__name__=~".*network.*", pod=~"frontend.*"}',
        '{__name__=~".*receive.*", pod=~"frontend.*"}',
        '{job="frontend"}',
        '{container="frontend"}',
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        try:
            params = {'query': query}
            response = requests.get(query_url, params=params)
            data = response.json()
            
            if data['status'] == 'success' and data['data']['result']:
                print(f"Found {len(data['data']['result'])} metrics:")
                for result in data['data']['result'][:10]:  # Show first 10
                    metric_name = result['metric']['__name__']
                    labels = {k: v for k, v in result['metric'].items() if k != '__name__'}
                    print(f"  - {metric_name} {labels}")
                if len(data['data']['result']) > 10:
                    print(f"  ... and {len(data['data']['result']) - 10} more")
            else:
                print("No results found")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    check_frontend_metrics()
