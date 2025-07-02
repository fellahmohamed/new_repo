#!/usr/bin/env python3
"""
Test script to verify that the GNN environment properly loads and uses the dataset
"""

import sys
import os
sys.path.append('/home/mohamed/Desktop/Pfe_new/gym-hpa-main')

import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_dataset_loading():
    """Test that the environment properly loads the dataset"""
    
    print("🧪 Testing Dataset Loading in GNN Environment...")
    
    try:
        from gym_hpa.envs import OnlineBoutiqueGNN
        
        # Create environment
        env = OnlineBoutiqueGNN(k8s=False, goal_reward='cost', 
                               waiting_period=0.1, use_gnn=True)
        
        print("✅ Environment created successfully")
        
        # Get dataset information
        dataset_info = env.get_dataset_info()
        print("\n📊 Dataset Information:")
        for key, value in dataset_info.items():
            print(f"   {key}: {value}")
        
        # Test environment reset and step
        print("\n🔄 Testing Environment Reset...")
        obs = env.reset()
        print(f"✅ Reset successful, observation shape: {obs.shape}")
        
        # Test a few steps to see if data is being used
        print("\n👟 Testing Environment Steps...")
        for step in range(3):
            action = env.action_space.sample()  # Random action
            obs, reward, done, info = env.step(action)
            
            print(f"   Step {step + 1}:")
            print(f"     Action: {action}")
            print(f"     Reward: {reward}")
            print(f"     Done: {done}")
            print(f"     GNN Enabled: {info.get('gnn_enabled', 'N/A')}")
            
            # Print some deployment metrics to verify data usage
            if step == 0:
                print(f"     Sample metrics from deployments:")
                for i, deployment in enumerate(env.deploymentList[:3]):  # Just first 3
                    print(f"       {deployment.name}: CPU={deployment.cpu_usage}, "
                          f"Memory={deployment.mem_usage}, "
                          f"Traffic In={deployment.received_traffic}")
            
            if done:
                break
        
        print("\n✅ All tests passed successfully!")
        
        # Test graph visualization
        print("\n🎨 Testing Graph Visualization...")
        try:
            env.visualize_graph()
        except ImportError:
            print("⚠️  Graph visualization requires matplotlib and networkx")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset_loading()
    if success:
        print("\n🎉 Dataset integration test completed successfully!")
    else:
        print("\n💥 Dataset integration test failed!")
        exit(1)
