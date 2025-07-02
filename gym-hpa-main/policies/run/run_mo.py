#!/usr/bin/env python3
"""
Multi-Objective Reinforcement Learning Runner for gym-hpa-MO

This script extends the gym-hpa project to support multi-objective optimization
for Kubernetes autoscaling, specifically optimizing both cost and latency.

Usage:
    python run_mo.py --algorithm ppo --env online-boutique-mo --episodes 1000 --weights 0.5,0.5
    python run_mo.py --algorithm a2c --env online-boutique-mo --episodes 500 --scalarization weighted --weights 0.7,0.3
    python run_mo.py --test --model-path models/mo_model.zip --episodes 100 --pareto-analysis
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import json

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import tensorboard

# Import gym-hpa environments
import gym_hpa


class MultiObjectiveCallback:
    """Callback for tracking multi-objective metrics during training"""
    
    def __init__(self, save_path: str, eval_freq: int = 1000):
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.episode_count = 0
        self.mo_metrics = {
            'episodes': [],
            'cost_rewards': [],
            'latency_rewards': [],
            'scalar_rewards': [],
            'pareto_solutions': []
        }
    
    def on_step(self, env, rewards: np.ndarray, dones: np.ndarray) -> bool:
        """Called at each environment step"""
        if hasattr(env, 'get_mo_info'):
            mo_info = env.get_mo_info()
            if mo_info and any(dones):
                self.episode_count += sum(dones)
                self.mo_metrics['episodes'].append(self.episode_count)
                self.mo_metrics['cost_rewards'].append(mo_info.get('cost_reward', 0))
                self.mo_metrics['latency_rewards'].append(mo_info.get('latency_reward', 0))
                self.mo_metrics['scalar_rewards'].append(mo_info.get('scalar_reward', 0))
                
                # Save metrics periodically
                if self.episode_count % self.eval_freq == 0:
                    self.save_metrics()
        
        return True
    
    def save_metrics(self):
        """Save metrics to file"""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(self.mo_metrics, f, indent=2)


class ParetoAnalyzer:
    """Utility class for Pareto front analysis"""
    
    @staticmethod
    def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
        """
        Find Pareto efficient solutions
        Args:
            costs: 2D array where each row is a solution [cost, latency]
        Returns:
            Boolean array indicating which solutions are Pareto efficient
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Remove dominated solutions (higher cost AND higher latency)
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
                is_efficient[i] = True
        return is_efficient
    
    @staticmethod
    def plot_pareto_front(solutions: List[Tuple[float, float]], 
                         pareto_solutions: List[Tuple[float, float]] = None,
                         save_path: str = None, title: str = "Pareto Front Analysis"):
        """Plot solutions and Pareto front"""
        if not solutions:
            print("No solutions to plot")
            return
        
        solutions = np.array(solutions)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(solutions[:, 0], solutions[:, 1], alpha=0.6, c='blue', label='All Solutions')
        
        if pareto_solutions is None:
            # Find Pareto front
            is_efficient = ParetoAnalyzer.is_pareto_efficient(solutions)
            pareto_solutions = solutions[is_efficient]
        else:
            pareto_solutions = np.array(pareto_solutions)
        
        if len(pareto_solutions) > 0:
            plt.scatter(pareto_solutions[:, 0], pareto_solutions[:, 1], 
                       c='red', s=100, label='Pareto Front', marker='*')
            
            # Sort for connecting line
            sorted_pareto = pareto_solutions[np.argsort(pareto_solutions[:, 0])]
            plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'r--', alpha=0.7)
        
        plt.xlabel('Cost (normalized)')
        plt.ylabel('Latency (normalized)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Pareto front plot saved to {save_path}")
        
        plt.show()


class MORunner:
    """Multi-Objective Reinforcement Learning Runner"""
    
    def __init__(self, args):
        self.args = args
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("results", f"mo_run_{self.timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Parse scalarization weights
        if args.weights:
            self.weights = [float(w) for w in args.weights.split(',')]
            if len(self.weights) != 2:
                raise ValueError("Exactly 2 weights required for cost and latency")
            if abs(sum(self.weights) - 1.0) > 1e-6:
                print(f"Warning: Weights sum to {sum(self.weights)}, normalizing...")
                total = sum(self.weights)
                self.weights = [w/total for w in self.weights]
        else:
            self.weights = [0.5, 0.5]  # Default equal weights
        
        print(f"Using weights: cost={self.weights[0]:.3f}, latency={self.weights[1]:.3f}")
        
    def create_environment(self, env_id: str = None):
        """Create and configure the multi-objective environment"""
        if env_id is None:
            env_id = self.args.env
        
        # Create environment with MO configuration
        env = gym.make(env_id)
        
        # Configure MO parameters
        if hasattr(env, 'set_mo_weights'):
            env.set_mo_weights(self.weights)
        
        if hasattr(env, 'set_scalarization'):
            env.set_scalarization(self.args.scalarization)
        
        # Monitor for logging
        log_dir = os.path.join(self.output_dir, "monitor")
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        
        return env
    
    def create_model(self, env):
        """Create RL model"""
        algorithm_map = {
            'ppo': PPO,
            'a2c': A2C,
            'dqn': DQN
        }
        
        if self.args.algorithm not in algorithm_map:
            raise ValueError(f"Unsupported algorithm: {self.args.algorithm}")
        
        model_class = algorithm_map[self.args.algorithm]
        
        # Model configuration
        model_kwargs = {
            'policy': 'MlpPolicy',
            'env': env,
            'verbose': 1,
            'tensorboard_log': os.path.join(self.output_dir, "tensorboard")
        }
        
        # Algorithm-specific parameters
        if self.args.algorithm == 'ppo':
            model_kwargs.update({
                'learning_rate': self.args.learning_rate,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01
            })
        elif self.args.algorithm == 'a2c':
            model_kwargs.update({
                'learning_rate': self.args.learning_rate,
                'n_steps': 5,
                'gamma': 0.99,
                'gae_lambda': 1.0,
                'ent_coef': 0.01,
                'vf_coef': 0.25
            })
        
        return model_class(**model_kwargs)
    
    def train(self):
        """Train the multi-objective RL agent"""
        print(f"Starting MO training with {self.args.algorithm.upper()}")
        print(f"Environment: {self.args.env}")
        print(f"Episodes: {self.args.episodes}")
        print(f"Scalarization: {self.args.scalarization}")
        
        # Create environment
        env = self.create_environment()
        
        # Create model
        model = self.create_model(env)
        
        # Setup callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=os.path.join(self.output_dir, "checkpoints"),
            name_prefix="mo_model"
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        eval_env = self.create_environment()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.output_dir, "best_model"),
            log_path=os.path.join(self.output_dir, "eval"),
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        # Train the model
        total_timesteps = self.args.episodes * 200  # Approximate timesteps per episode
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=f"mo_{self.args.algorithm}_{self.timestamp}"
        )
        
        # Save final model
        model_path = os.path.join(self.output_dir, "final_model.zip")
        model.save(model_path)
        
        print(f"Training completed. Model saved to {model_path}")
        return model_path
    
    def test(self):
        """Test the trained model and perform Pareto analysis"""
        print(f"Testing model: {self.args.model_path}")
        
        # Load model
        if self.args.algorithm == 'ppo':
            model = PPO.load(self.args.model_path)
        elif self.args.algorithm == 'a2c':
            model = A2C.load(self.args.model_path)
        elif self.args.algorithm == 'dqn':
            model = DQN.load(self.args.model_path)
        
        # Create environment
        env = self.create_environment()
        
        # Test episodes
        solutions = []
        episode_rewards = []
        
        for episode in range(self.args.test_episodes):
            obs = env.reset()
            episode_reward = 0
            cost_sum = 0
            latency_sum = 0
            steps = 0
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                # Collect MO metrics
                if hasattr(env, 'get_mo_info'):
                    mo_info = env.get_mo_info()
                    if mo_info:
                        cost_sum += mo_info.get('cost_reward', 0)
                        latency_sum += mo_info.get('latency_reward', 0)
                
                steps += 1
                if done:
                    break
            
            # Store solution
            if steps > 0:
                avg_cost = cost_sum / steps
                avg_latency = latency_sum / steps
                solutions.append((avg_cost, avg_latency))
                episode_rewards.append(episode_reward)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{self.args.test_episodes} completed")
        
        # Analysis
        print(f"\nTest Results:")
        print(f"Average episode reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
        
        if solutions:
            solutions_array = np.array(solutions)
            print(f"Average cost: {np.mean(solutions_array[:, 0]):.3f}")
            print(f"Average latency: {np.mean(solutions_array[:, 1]):.3f}")
            
            # Pareto analysis
            if self.args.pareto_analysis:
                print("\nPerforming Pareto analysis...")
                analyzer = ParetoAnalyzer()
                plot_path = os.path.join(self.output_dir, "pareto_front.png")
                analyzer.plot_pareto_front(
                    solutions, 
                    save_path=plot_path,
                    title=f"Pareto Front - {self.args.algorithm.upper()} (Test)"
                )
                
                # Find Pareto efficient solutions
                is_efficient = analyzer.is_pareto_efficient(solutions_array)
                pareto_solutions = solutions_array[is_efficient]
                print(f"Found {len(pareto_solutions)} Pareto efficient solutions out of {len(solutions)}")
        
        return solutions, episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Multi-Objective Reinforcement Learning for Kubernetes Autoscaling")
    
    # Mode selection
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--model-path', type=str, help='Path to trained model (required for test mode)')
    
    # Training parameters
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'a2c', 'dqn'],
                       help='RL algorithm to use')
    parser.add_argument('--env', type=str, default='online-boutique-mo-v0',
                       help='Environment ID')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--test-episodes', type=int, default=100,
                       help='Number of test episodes')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    
    # Multi-objective parameters
    parser.add_argument('--scalarization', type=str, default='weighted', 
                       choices=['weighted', 'tchebycheff', 'epsilon'],
                       help='Scalarization method')
    parser.add_argument('--weights', type=str, default='0.5,0.5',
                       help='Comma-separated weights for cost,latency')
    
    # Analysis options
    parser.add_argument('--pareto-analysis', action='store_true',
                       help='Perform Pareto front analysis')
    parser.add_argument('--save-results', action='store_true', default=True,
                       help='Save results and plots')
    
    args = parser.parse_args()
    
    # Validation
    if args.test and not args.model_path:
        parser.error("--model-path is required when using --test mode")
    
    # Run
    runner = MORunner(args)
    
    if args.test:
        solutions, rewards = runner.test()
    else:
        model_path = runner.train()
        print(f"\nTraining completed!")
        print(f"Model saved to: {model_path}")
        print(f"Results saved to: {runner.output_dir}")
        print(f"\nTo test the model, run:")
        print(f"python run_mo.py --test --model-path {model_path} --pareto-analysis")


if __name__ == "__main__":
    main()
