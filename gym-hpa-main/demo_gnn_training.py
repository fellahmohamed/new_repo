#!/usr/bin/env python3
"""
Demo script showing how to train a RL agent on the GNN-enhanced environment
"""

import sys
import os
sys.path.append('/home/mohamed/Desktop/Pfe_new/gym-hpa-main')

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import logging

def train_gnn_agent():
    """Train a PPO agent on the GNN-enhanced environment"""
    
    print("🏋️‍♂️ Training RL Agent on GNN Environment...")
    
    try:
        from gym_hpa.envs import OnlineBoutiqueGNN
        
        # Create environment
        env = OnlineBoutiqueGNN(k8s=False, goal_reward='cost', 
                               waiting_period=0.1, use_gnn=True)
        
        # Wrap in vectorized environment
        env = DummyVecEnv([lambda: env])
        
        print(f"✅ Environment created with observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        
        # Create PPO model
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            n_steps=128,
            batch_size=64,
            learning_rate=3e-4,
            tensorboard_log="./gnn_tensorboard/"
        )
        
        print("✅ PPO model created")
        
        # Create callback for saving
        checkpoint_callback = CheckpointCallback(
            save_freq=500, 
            save_path="./gnn_models/", 
            name_prefix="gnn_ppo"
        )
        
        # Train the model
        print("🚀 Starting training...")
        model.learn(
            total_timesteps=2000, 
            tb_log_name="gnn_training",
            callback=checkpoint_callback
        )
        
        # Save final model
        model.save("gnn_ppo_final")
        print("✅ Training completed and model saved!")
        
        return model, env
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def evaluate_gnn_agent(model, env, n_episodes=5):
    """Evaluate the trained agent"""
    
    print(f"\\n📊 Evaluating agent for {n_episodes} episodes...")
    
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < 25:  # Max 25 steps per episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Since we're using vectorized env, extract values
            if hasattr(reward, '__len__') and len(reward) > 0:
                episode_reward += reward[0]
            else:
                episode_reward += reward
            
            step_count += 1
        
        episode_rewards.append(episode_reward)
        print(f"   Episode {episode + 1}: Reward={episode_reward:.3f}")
    
    print(f"\\n📈 Evaluation Results:")
    print(f"   Average Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    
    return episode_rewards

def compare_gnn_vs_standard_training():
    """Compare training on GNN vs standard environment"""
    
    print("\\n⚔️  Comparing GNN vs Standard Environment Training...")
    
    try:
        from gym_hpa.envs import OnlineBoutiqueGNN
        
        # Create environments (both using 'cost' goal for scalar rewards)
        env_gnn = DummyVecEnv([lambda: OnlineBoutiqueGNN(k8s=False, use_gnn=True, goal_reward='cost', waiting_period=0.1)])
        env_standard = DummyVecEnv([lambda: OnlineBoutiqueGNN(k8s=False, use_gnn=False, goal_reward='cost', waiting_period=0.1)])
        
        print(f"   GNN obs space: {env_gnn.observation_space}")
        print(f"   Standard obs space: {env_standard.observation_space}")
        
        # Create models
        model_gnn = PPO("MlpPolicy", env_gnn, verbose=0, n_steps=64)
        model_standard = PPO("MlpPolicy", env_standard, verbose=0, n_steps=64)
        
        # Short training
        print("   Training GNN model...")
        model_gnn.learn(total_timesteps=500)
        
        print("   Training Standard model...")
        model_standard.learn(total_timesteps=500)
        
        print("✅ Comparison training completed")
        
        # Quick evaluation
        print("\\n   Quick evaluation (2 episodes each):")
        
        # Evaluate GNN
        gnn_rewards = []
        for _ in range(2):
            obs = env_gnn.reset()
            total_reward = 0
            for _ in range(10):
                action, _ = model_gnn.predict(obs, deterministic=True)
                obs, reward, done, info = env_gnn.step(action)
                total_reward += reward[0]
            gnn_rewards.append(total_reward)
        
        # Evaluate Standard
        standard_rewards = []
        for _ in range(2):
            obs = env_standard.reset()
            total_reward = 0
            for _ in range(10):
                action, _ = model_standard.predict(obs, deterministic=True)
                obs, reward, done, info = env_standard.step(action)
                total_reward += reward[0]
            standard_rewards.append(total_reward)
        
        print(f"   GNN average: {np.mean(gnn_rewards):.3f}")
        print(f"   Standard average: {np.mean(standard_rewards):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demo function"""
    
    print("🎯 GNN Environment Training Demo\\n")
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    # Train agent
    model, env = train_gnn_agent()
    
    if model is not None:
        # Evaluate agent
        evaluate_gnn_agent(model, env)
        
        # Compare with standard environment
        compare_gnn_vs_standard_training()
        
        print("\\n🎉 Demo completed successfully!")
        print("\\n📁 Files created:")
        print("   - gnn_ppo_final.zip (trained model)")
        print("   - gnn_models/ (checkpoint models)")
        print("   - gnn_tensorboard/ (training logs)")
        print("   - microservice_graph_gnn.png (graph visualization)")
        
        print("\\n🔧 Next steps:")
        print("   1. Run longer training: --total_steps 10000")
        print("   2. Experiment with different GNN architectures")
        print("   3. Compare with multi-objective algorithms")
        print("   4. Analyze microservice dependency impacts")
        
    else:
        print("\\n❌ Demo failed. Please check the implementation.")

if __name__ == "__main__":
    main()
