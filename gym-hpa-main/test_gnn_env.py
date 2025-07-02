#!/usr/bin/env python3
"""
Test script for the GNN-enhanced OnlineBoutique environment
"""

import sys
import os
sys.path.append('/home/mohamed/Desktop/Pfe_new/gym-hpa-main')

import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx

def test_gnn_environment():
    """Test the GNN environment functionality"""
    
    print("🧪 Testing GNN Environment...")
    
    try:
        from gym_hpa.envs import OnlineBoutiqueGNN
        print("✅ OnlineBoutiqueGNN imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        # Test environment creation with GNN enabled
        env_gnn = OnlineBoutiqueGNN(k8s=False, goal_reward='cost_latency', 
                                   waiting_period=0.1, use_gnn=True)
        print("✅ GNN environment created successfully")
        
        # Test environment creation with GNN disabled
        env_standard = OnlineBoutiqueGNN(k8s=False, goal_reward='cost_latency', 
                                        waiting_period=0.1, use_gnn=False)
        print("✅ Standard environment created successfully")
        
    except Exception as e:
        print(f"❌ Environment creation error: {e}")
        return False
    
    try:
        # Test reset
        obs_gnn = env_gnn.reset()
        obs_standard = env_standard.reset()
        print(f"✅ Environment reset successful")
        print(f"   GNN observation shape: {obs_gnn.shape}")
        print(f"   Standard observation shape: {obs_standard.shape}")
        
    except Exception as e:
        print(f"❌ Reset error: {e}")
        return False
    
    try:
        # Test step with dummy action
        action = [0, 0]  # Do nothing action
        obs_gnn, reward_gnn, done_gnn, info_gnn = env_gnn.step(action)
        obs_standard, reward_standard, done_standard, info_standard = env_standard.step(action)
        
        print(f"✅ Environment step completed")
        print(f"   GNN reward: {reward_gnn}")
        print(f"   Standard reward: {reward_standard}")
        print(f"   GNN enabled in info: {info_gnn.get('gnn_enabled', False)}")
        
    except Exception as e:
        print(f"❌ Step error: {e}")
        return False
    
    try:
        # Test graph information
        graph_info = env_gnn.get_graph_info()
        print(f"✅ Graph info retrieved")
        print(f"   Number of services: {graph_info['num_services']}")
        print(f"   Adjacency matrix shape: {graph_info['adjacency_matrix'].shape}")
        print(f"   Services: {graph_info['service_names'][:3]}...")  # Show first 3
        
    except Exception as e:
        print(f"❌ Graph info error: {e}")
        return False
    
    try:
        # Test GNN model directly
        print("\n🧠 Testing GNN Model...")
        
        # Get the graph and features
        node_features = env_gnn.graph.get_node_features(env_gnn.deploymentList)
        adjacency_matrix = env_gnn.graph.adjacency_matrix
        
        print(f"   Node features shape: {node_features.shape}")
        print(f"   Adjacency matrix shape: {adjacency_matrix.shape}")
        
        # Forward pass through GNN
        with torch.no_grad():
            gnn_output = env_gnn.gnn(node_features, adjacency_matrix)
        print(f"   GNN output shape: {gnn_output.shape}")
        print(f"✅ GNN forward pass successful")
        
    except Exception as e:
        print(f"❌ GNN model error: {e}")
        return False
    
    return True

def visualize_microservice_graph():
    """Visualize the microservice dependency graph"""
    
    print("\n📊 Creating Microservice Graph Visualization...")
    
    try:
        from gym_hpa.envs import OnlineBoutiqueGNN
        
        # Create environment
        env = OnlineBoutiqueGNN(k8s=False, use_gnn=True)
        
        # Get graph info
        graph_info = env.get_graph_info()
        adj_matrix = graph_info['adjacency_matrix']
        service_names = graph_info['service_names']
        
        # Create networkx graph
        G = nx.from_numpy_array(adj_matrix)
        
        # Create labels with shorter names for better visualization
        labels = {}
        short_names = {
            'recommendationservice': 'recommendation',
            'productcatalogservice': 'catalog', 
            'cartservice': 'cart',
            'adservice': 'ads',
            'paymentservice': 'payment',
            'shippingservice': 'shipping', 
            'currencyservice': 'currency',
            'redis-cart': 'redis',
            'checkoutservice': 'checkout',
            'frontend': 'frontend',
            'emailservice': 'email'
        }
        
        for i, name in enumerate(service_names):
            labels[i] = short_names.get(name, name)
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=2000, alpha=0.8)
        
        # Draw edges with different weights
        edges = G.edges()
        weights = [G[u][v]['weight'] if 'weight' in G[u][v] else 1.0 for u, v in edges]
        nx.draw_networkx_edges(G, pos, alpha=0.6, width=weights)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        plt.title("Online Boutique Microservice Dependency Graph\\n(GNN-Enhanced Environment)", 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('/home/mohamed/Desktop/Pfe_new/gym-hpa-main/microservice_graph_gnn.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Graph visualization saved as 'microservice_graph_gnn.png'")
        
        # Print adjacency matrix info
        print(f"\\n📋 Graph Statistics:")
        print(f"   Number of nodes: {G.number_of_nodes()}")
        print(f"   Number of edges: {G.number_of_edges()}")
        print(f"   Graph density: {nx.density(G):.3f}")
        print(f"   Is connected: {nx.is_connected(G.to_undirected())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_gnn_vs_standard():
    """Compare GNN-enhanced vs standard environment performance"""
    
    print("\\n⚖️  Comparing GNN vs Standard Environment...")
    
    try:
        from gym_hpa.envs import OnlineBoutiqueGNN
        
        # Create both environments
        env_gnn = OnlineBoutiqueGNN(k8s=False, use_gnn=True)
        env_standard = OnlineBoutiqueGNN(k8s=False, use_gnn=False)
        
        # Reset both
        obs_gnn = env_gnn.reset()
        obs_standard = env_standard.reset()
        
        print(f"   GNN obs dim: {len(obs_gnn)} vs Standard obs dim: {len(obs_standard)}")
        
        # Run a few steps
        rewards_gnn = []
        rewards_standard = []
        
        for step in range(5):
            # Random actions
            action = [np.random.randint(0, 11), np.random.randint(0, 15)]
            
            obs_gnn, reward_gnn, done_gnn, info_gnn = env_gnn.step(action)
            obs_standard, reward_standard, done_standard, info_standard = env_standard.step(action)
            
            rewards_gnn.append(reward_gnn)
            rewards_standard.append(reward_standard)
        
        print(f"   Average GNN reward: {np.mean(rewards_gnn, axis=0)}")
        print(f"   Average Standard reward: {np.mean(rewards_standard, axis=0)}")
        print("✅ Comparison completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison error: {e}")
        return False

def main():
    """Main test function"""
    
    print("🚀 Starting GNN Environment Tests...\\n")
    
    # Run tests
    success = True
    
    success &= test_gnn_environment()
    success &= visualize_microservice_graph()  
    success &= compare_gnn_vs_standard()
    
    if success:
        print("\\n🎉 All tests passed! GNN environment is ready to use.")
        print("\\n📝 Usage Example:")
        print("   python policies/run/run.py --use_case online_boutique_gnn --alg ppo --training --total_steps 1000")
    else:
        print("\\n❌ Some tests failed. Please check the implementation.")
    
    return success

if __name__ == "__main__":
    main()
