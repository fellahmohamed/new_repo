#!/usr/bin/env python3
"""
Quick GNN training demo with real data
"""

import sys
sys.path.append('/home/mohamed/Desktop/Pfe_new/gym-hpa-main')

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import logging

logging.basicConfig(level=logging.INFO)

def quick_train():
    """Quick training run"""
    print("🚀 Quick GNN Training with Real Data...")
    
    from gym_hpa.envs import OnlineBoutiqueGNN
    
    # Create environment with scalar rewards for standard RL
    env = OnlineBoutiqueGNN(k8s=False, use_gnn=True, goal_reward='cost')
    
    # Check dataset
    dataset_info = env.get_dataset_info()
    print(f"Dataset loaded: {dataset_info['dataset_loaded']}")
    print(f"Dataset shape: {dataset_info['dataset_shape']}")
    
    # Vectorize environment
    env = DummyVecEnv([lambda: env])
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create PPO model
    model = PPO("MlpPolicy", env, verbose=1, n_steps=512, batch_size=32)
    
    print("✅ Starting training...")
    
    # Train for fewer timesteps for quick demo
    model.learn(total_timesteps=5000)
    
    print("✅ Training completed!")
    
    # Save model
    model.save("gnn_quick_demo")
    print("💾 Model saved as 'gnn_quick_demo'")
    
    # Quick evaluation
    print("\n🧪 Quick evaluation...")
    obs = env.reset()
    for i in range(5):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: Reward={reward}, Done={done}")
        if done:
            obs = env.reset()
    
    print("✅ Demo completed successfully!")

if __name__ == "__main__":
    quick_train()
