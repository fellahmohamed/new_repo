#!/usr/bin/env python3
"""
Compare the behavior when using real dataset vs random data
"""

import sys
import os
sys.path.append('/home/mohamed/Desktop/Pfe_new/gym-hpa-main')

import numpy as np
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Reduce verbose output

def compare_data_sources():
    """Compare environment behavior with real dataset vs random data"""
    
    print("📊 Comparing Real Dataset vs Random Data...")
    
    try:
        from gym_hpa.envs import OnlineBoutiqueGNN
        
        # Test with real dataset
        print("\n1️⃣  Testing with Real Dataset...")
        env_real = OnlineBoutiqueGNN(k8s=False, goal_reward='cost', 
                                    waiting_period=0.1, use_gnn=True)
        
        dataset_info = env_real.get_dataset_info()
        print(f"   Dataset loaded: {dataset_info['dataset_loaded']}")
        print(f"   Sample count: {dataset_info['sample_count']}")
        print(f"   Date range: {dataset_info['date_range']['start']} to {dataset_info['date_range']['end']}")
        
        # Collect metrics from real data
        real_metrics = []
        obs = env_real.reset()
        for _ in range(5):
            action = [0, 0]  # Do nothing action
            obs, reward, done, info = env_real.step(action)
            
            # Collect CPU usage from all services
            cpu_usage = [d.cpu_usage for d in env_real.deploymentList]
            real_metrics.append(cpu_usage)
        
        # Test with forced random data (by providing empty dataframe)
        print("\n2️⃣  Testing with Random Data...")
        env_random = OnlineBoutiqueGNN(k8s=False, goal_reward='cost', 
                                      waiting_period=0.1, use_gnn=True)
        
        # Force empty dataframe to use random data
        import pandas as pd
        env_random.df = pd.DataFrame()
        
        # Collect metrics from random data
        random_metrics = []
        obs = env_random.reset()
        for _ in range(5):
            action = [0, 0]  # Do nothing action
            obs, reward, done, info = env_random.step(action)
            
            # Collect CPU usage from all services
            cpu_usage = [d.cpu_usage for d in env_random.deploymentList]
            random_metrics.append(cpu_usage)
        
        # Compare the metrics
        print("\n3️⃣  Comparison Results:")
        print("   Real Data CPU Usage Ranges:")
        real_array = np.array(real_metrics)
        for i, service in enumerate(env_real.deploymentList):
            service_cpu = real_array[:, i]
            print(f"     {service.name}: {service_cpu.min():.1f} - {service_cpu.max():.1f} (avg: {service_cpu.mean():.1f})")
        
        print("\n   Random Data CPU Usage Ranges:")
        random_array = np.array(random_metrics)
        for i, service in enumerate(env_random.deploymentList):
            service_cpu = random_array[:, i]
            print(f"     {service.name}: {service_cpu.min():.1f} - {service_cpu.max():.1f} (avg: {service_cpu.mean():.1f})")
        
        # Create visualization
        print("\n4️⃣  Creating Comparison Visualization...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot real data
        services = [d.name[:10] for d in env_real.deploymentList]  # Truncate names
        ax1.boxplot(real_array, labels=services)
        ax1.set_title('CPU Usage Distribution - Real Dataset')
        ax1.set_ylabel('CPU Usage')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot random data
        ax2.boxplot(random_array, labels=services)
        ax2.set_title('CPU Usage Distribution - Random Data')
        ax2.set_ylabel('CPU Usage')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('data_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Comparison visualization saved as 'data_comparison.png'")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = compare_data_sources()
    if success:
        print("\n🎉 Data source comparison completed successfully!")
        print("   The environment now uses real historical data from the CSV file")
        print("   This provides more realistic microservice behavior patterns")
    else:
        print("\n💥 Data source comparison failed!")
        exit(1)
