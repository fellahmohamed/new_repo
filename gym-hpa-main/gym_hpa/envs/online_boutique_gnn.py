import csv
import datetime
from datetime import datetime
import logging
import time
from statistics import mean

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
import torch
import torch.nn as nn
import torch.nn.functional as F

# Number of Requests - Discrete Event
from gym_hpa.envs.deployment import get_max_cpu, get_max_mem, get_max_traffic, get_online_boutique_deployment_list
from gym_hpa.envs.util import save_to_csv, get_num_pods, get_cost_reward, \
    get_latency_reward_online_boutique

# MIN and MAX Replication
MIN_REPLICATION = 1
MAX_REPLICATION = 8

MAX_STEPS = 25  # MAX Number of steps per episode

# Possible Actions (Discrete)
ACTION_DO_NOTHING = 0
ACTION_ADD_1_REPLICA = 1
ACTION_ADD_2_REPLICA = 2
ACTION_ADD_3_REPLICA = 3
ACTION_ADD_4_REPLICA = 4
ACTION_ADD_5_REPLICA = 5
ACTION_ADD_6_REPLICA = 6
ACTION_ADD_7_REPLICA = 7
ACTION_TERMINATE_1_REPLICA = 8
ACTION_TERMINATE_2_REPLICA = 9
ACTION_TERMINATE_3_REPLICA = 10
ACTION_TERMINATE_4_REPLICA = 11
ACTION_TERMINATE_5_REPLICA = 12
ACTION_TERMINATE_6_REPLICA = 13
ACTION_TERMINATE_7_REPLICA = 14

# Deployments
DEPLOYMENTS = ["recommendationservice", "productcatalogservice", "cartservice", "adservice",
               "paymentservice", "shippingservice", "currencyservice", "redis-cart",
               "checkoutservice", "frontend", "emailservice"]

# Action Moves
MOVES = ["None", "Add-1", "Add-2", "Add-3", "Add-4", "Add-5", "Add-6", "Add-7",
         "Stop-1", "Stop-2", "Stop-3", "Stop-4", "Stop-5", "Stop-6", "Stop-7"]

# IDs
ID_DEPLOYMENTS = 0
ID_MOVES = 1

ID_recommendation = 0
ID_product_catalog = 1
ID_cart_service = 2
ID_ad_service = 3
ID_payment_service = 4
ID_shipping_service = 5
ID_currency_service = 6
ID_redis_cart = 7
ID_checkout_service = 8
ID_frontend = 9
ID_email = 10

# Reward objectives
LATENCY = 'latency'
COST = 'cost'


class SimpleGCN(nn.Module):
    """
    Simple Graph Convolutional Network for microservice dependency modeling
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, node_features, adjacency_matrix):
        """
        Forward pass through GCN
        Args:
            node_features: (num_nodes, input_dim) - Features for each microservice
            adjacency_matrix: (num_nodes, num_nodes) - Dependencies between services
        Returns:
            output: (num_nodes, output_dim) - Enhanced node features
        """
        # First GCN layer: Aggregate neighbor information
        x = torch.mm(adjacency_matrix, node_features)  # Message passing
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = torch.mm(adjacency_matrix, x)  # Message passing
        x = self.fc2(x)
        
        return x


class MicroserviceGraph:
    """
    Models the microservice dependency graph for Online Boutique
    """
    def __init__(self):
        self.num_services = 11
        self.service_names = DEPLOYMENTS
        
        # Create adjacency matrix based on Online Boutique architecture
        self.adjacency_matrix = self._create_adjacency_matrix()
        
    def _create_adjacency_matrix(self):
        """
        Create adjacency matrix representing microservice dependencies
        Based on Online Boutique architecture:
        - Frontend calls: recommendation, product catalog, cart, ad, checkout, currency, shipping
        - Checkout calls: payment, shipping, email, cart, currency
        - Cart calls: redis-cart
        - Recommendation calls: product catalog
        """
        adj = np.zeros((self.num_services, self.num_services))
        
        # Frontend dependencies (frontend calls these services)
        frontend_deps = [ID_recommendation, ID_product_catalog, ID_cart_service, 
                        ID_ad_service, ID_checkout_service, ID_currency_service, ID_shipping_service]
        for dep in frontend_deps:
            adj[ID_frontend, dep] = 1
            adj[dep, ID_frontend] = 1  # Bidirectional for GCN
            
        # Checkout service dependencies
        checkout_deps = [ID_payment_service, ID_shipping_service, ID_email, 
                        ID_cart_service, ID_currency_service]
        for dep in checkout_deps:
            adj[ID_checkout_service, dep] = 1
            adj[dep, ID_checkout_service] = 1
            
        # Cart service dependencies
        adj[ID_cart_service, ID_redis_cart] = 1
        adj[ID_redis_cart, ID_cart_service] = 1
        
        # Recommendation service dependencies
        adj[ID_recommendation, ID_product_catalog] = 1
        adj[ID_product_catalog, ID_recommendation] = 1
        
        # Add self-loops for better GCN performance
        np.fill_diagonal(adj, 1)
        
        # Normalize adjacency matrix (simple row normalization)
        row_sum = adj.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1  # Avoid division by zero
        adj = adj / row_sum
        
        return torch.FloatTensor(adj)
    
    def get_node_features(self, deployment_list):
        """
        Extract node features for each microservice
        Returns tensor of shape (num_services, feature_dim)
        """
        features = []
        for i, deployment in enumerate(deployment_list):
            feature_vector = [
                deployment.num_pods / MAX_REPLICATION,  # Normalized pod count
                deployment.desired_replicas / MAX_REPLICATION,  # Normalized desired replicas
                deployment.cpu_usage / get_max_cpu(),  # Normalized CPU usage
                deployment.mem_usage / get_max_mem(),  # Normalized memory usage
                deployment.received_traffic / get_max_traffic(),  # Normalized incoming traffic
                deployment.transmit_traffic / get_max_traffic(),  # Normalized outgoing traffic
                deployment.latency / 1000.0 if hasattr(deployment, 'latency') else 0.0,  # Normalized latency
            ]
            features.append(feature_vector)
        
        return torch.FloatTensor(features)


class OnlineBoutiqueGNN(gym.Env):
    """
    Horizontal Scaling for Online Boutique in Kubernetes using Graph Neural Networks
    """

    metadata = {'render.modes': ['human', 'ansi', 'array']}

    def __init__(self, k8s=False, goal_reward="cost_latency", waiting_period=0.3, use_gnn=True):
        super(OnlineBoutiqueGNN, self).__init__()

        self.k8s = k8s
        self.name = "online_boutique_gnn"
        self.__version__ = "0.0.1"
        self.seed()
        self.goal_reward = goal_reward
        self.waiting_period = waiting_period
        self.use_gnn = use_gnn

        logging.info("[Init] GNN Env: {} | K8s: {} | Version {} | GNN: {}".format(
            self.name, self.k8s, self.__version__, self.use_gnn))

        # Current Step
        self.current_step = 0

        # Actions identified by integers 0-n -> 15 actions!
        self.num_actions = 15
        self.action_space = spaces.MultiDiscrete([11, self.num_actions])

        self.min_pods = MIN_REPLICATION
        self.max_pods = MAX_REPLICATION
        self.num_apps = 11

        # Initialize microservice graph
        self.graph = MicroserviceGraph()
        
        # Initialize GNN if enabled
        if self.use_gnn:
            self.node_feature_dim = 7  # Features per microservice
            self.gnn_hidden_dim = 32
            self.gnn_output_dim = 16
            self.gnn = SimpleGCN(self.node_feature_dim, self.gnn_hidden_dim, self.gnn_output_dim)
            
            # Observation space includes both raw features and GNN embeddings
            obs_dim = self.num_apps * (self.node_feature_dim + self.gnn_output_dim)
        else:
            # Standard observation space
            obs_dim = self.num_apps * 6  # 6 metrics per service

        # Deployment Data
        self.deploymentList = get_online_boutique_deployment_list(self.k8s, self.min_pods, self.max_pods)

        # Logging Deployment
        for d in self.deploymentList:
            d.print_deployment()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Info
        self.total_cost_reward = 0
        self.total_latency_reward = 0
        self.avg_pods = []
        self.avg_latency = []
        self.avg_cost_rewards = []
        self.avg_latency_rewards = []

        # episode over
        self.episode_over = False
        self.info = {}

        # Keywords for Reward calculation
        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False

        self.time_start = 0
        self.execution_time = 0
        self.episode_count = 0
        self.file_results = "results.csv"
        self.obs_csv = self.name + "_observation.csv"
        
        # Try to load simulation data if available
        try:
            # Use the correct absolute path to the dataset
            dataset_path = "/home/mohamed/Desktop/Pfe_new/gym-hpa-main/datasets/real/default/v1/online_boutique_gym_observation.csv"
            self.df = pd.read_csv(dataset_path)
            logging.info(f"Successfully loaded simulation data from {dataset_path}")
            logging.info(f"Dataset shape: {self.df.shape}")
        except Exception as e:
            # Create dummy dataframe if file doesn't exist
            self.df = pd.DataFrame()
            logging.warning(f"Could not load simulation data: {e}. Using random data instead")

    def step(self, action):
        if self.current_step == 1:
            if not self.k8s:
                self.simulation_update()
            self.time_start = time.time()

        # Get first action: deployment
        if action[ID_DEPLOYMENTS] == 0:  # recommendation
            n = ID_recommendation
        elif action[ID_DEPLOYMENTS] == 1:  # product catalog
            n = ID_product_catalog
        elif action[ID_DEPLOYMENTS] == 2:  # cart_service
            n = ID_cart_service
        elif action[ID_DEPLOYMENTS] == 3:  # ad_service
            n = ID_ad_service
        elif action[ID_DEPLOYMENTS] == 4:  # payment_service
            n = ID_payment_service
        elif action[ID_DEPLOYMENTS] == 5:  # shipping_service
            n = ID_shipping_service
        elif action[ID_DEPLOYMENTS] == 6:  # currency_service
            n = ID_currency_service
        elif action[ID_DEPLOYMENTS] == 7:  # redis_cart
            n = ID_redis_cart
        elif action[ID_DEPLOYMENTS] == 8:  # checkout_service
            n = ID_checkout_service
        elif action[ID_DEPLOYMENTS] == 9:  # frontend
            n = ID_frontend
        else:  # ==10 email
            n = ID_email

        # Execute one time step within the environment
        self.take_action(action[ID_MOVES], n)

        # Wait a few seconds if on real k8s cluster
        if self.k8s:
            if action[ID_MOVES] != ACTION_DO_NOTHING \
                    and self.constraint_min_pod_replicas is False \
                    and self.constraint_max_pod_replicas is False:
                time.sleep(self.waiting_period)

        # Update observation before reward calculation:
        if self.k8s:
            for d in self.deploymentList:
                d.update_obs_k8s()
        else:
            self.simulation_update()

        # Get reward
        reward = self.get_reward
        self.total_cost_reward += reward[0]
        self.total_latency_reward += reward[1]
        self.avg_cost_rewards.append(reward[0])
        self.avg_latency_rewards.append(reward[1])

        self.avg_pods.append(get_num_pods(self.deploymentList))
        self.avg_latency.append(self.deploymentList[0].latency)

        logging.info('[Step {}] | Action (Deployment): {} | Action (Move): {} | Vector Reward: {} | Total Cost Reward: {} | Total Latency Reward: {}'.format(
            self.current_step, DEPLOYMENTS[action[0]], MOVES[action[1]], reward, self.total_cost_reward, self.total_latency_reward))

        ob = self.get_state()
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.info = dict(
            total_cost_reward=self.total_cost_reward,
            total_latency_reward=self.total_latency_reward,
            cost_reward=reward[0],
            latency_reward=reward[1],
            gnn_enabled=self.use_gnn
        )

        # Update Reward Keywords
        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False

        if self.current_step == MAX_STEPS:
            self.episode_count += 1
            self.execution_time = time.time() - self.time_start

        # Return vector rewards for multi-objective RL or scalarized reward for standard RL
        if self.goal_reward == "cost_latency":
            # Return vector reward for multi-objective RL
            return np.array(ob), np.array(reward), self.episode_over, self.info
        else:
            # Return scalarized reward for standard RL
            scalarized_reward = self.scalarize_reward(reward)
            return np.array(ob), scalarized_reward, self.episode_over, self.info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        """
        self.current_step = 0
        self.episode_over = False
        self.total_cost_reward = 0
        self.total_latency_reward = 0
        self.avg_cost_rewards = []
        self.avg_latency_rewards = []
        self.avg_pods = []
        self.avg_latency = []

        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False

        # Deployment Data
        self.deploymentList = get_online_boutique_deployment_list(self.k8s, self.min_pods, self.max_pods)

        return np.array(self.get_state())

    def render(self, mode='human', close=False):
        return

    def take_action(self, action, id):
        self.current_step += 1

        # Stop if MAX_STEPS
        if self.current_step == MAX_STEPS:
            self.episode_over = True

        # ACTIONS
        if action == ACTION_DO_NOTHING:
            pass
        elif action == ACTION_ADD_1_REPLICA:
            self.deploymentList[id].deploy_pod_replicas(1, self)
        elif action == ACTION_ADD_2_REPLICA:
            self.deploymentList[id].deploy_pod_replicas(2, self)
        elif action == ACTION_ADD_3_REPLICA:
            self.deploymentList[id].deploy_pod_replicas(3, self)
        elif action == ACTION_ADD_4_REPLICA:
            self.deploymentList[id].deploy_pod_replicas(4, self)
        elif action == ACTION_ADD_5_REPLICA:
            self.deploymentList[id].deploy_pod_replicas(5, self)
        elif action == ACTION_ADD_6_REPLICA:
            self.deploymentList[id].deploy_pod_replicas(6, self)
        elif action == ACTION_ADD_7_REPLICA:
            self.deploymentList[id].deploy_pod_replicas(7, self)
        elif action == ACTION_TERMINATE_1_REPLICA:
            self.deploymentList[id].terminate_pod_replicas(1, self)
        elif action == ACTION_TERMINATE_2_REPLICA:
            self.deploymentList[id].terminate_pod_replicas(2, self)
        elif action == ACTION_TERMINATE_3_REPLICA:
            self.deploymentList[id].terminate_pod_replicas(3, self)
        elif action == ACTION_TERMINATE_4_REPLICA:
            self.deploymentList[id].terminate_pod_replicas(4, self)
        elif action == ACTION_TERMINATE_5_REPLICA:
            self.deploymentList[id].terminate_pod_replicas(5, self)
        elif action == ACTION_TERMINATE_6_REPLICA:
            self.deploymentList[id].terminate_pod_replicas(6, self)
        elif action == ACTION_TERMINATE_7_REPLICA:
            self.deploymentList[id].terminate_pod_replicas(7, self)
        else:
            logging.info('[Take Action] Unrecognized Action: ' + str(action))

    @property
    def get_reward(self):
        """Calculate Rewards"""
        if self.constraint_max_pod_replicas or self.constraint_min_pod_replicas:
            return [-1, -3000]
        
        reward = self.calculate_reward()
        return reward

    def get_state(self):
        """
        Get observation state, potentially enhanced with GNN embeddings
        """
        if self.use_gnn:
            return self.get_gnn_enhanced_state()
        else:
            return self.get_standard_state()
    
    def get_standard_state(self):
        """Standard observation without GNN"""
        obs = []
        for deployment in self.deploymentList:
            obs.extend([
                deployment.num_pods / MAX_REPLICATION,
                deployment.desired_replicas / MAX_REPLICATION,
                deployment.cpu_usage / get_max_cpu(),
                deployment.mem_usage / get_max_mem(),
                deployment.received_traffic / get_max_traffic(),
                deployment.transmit_traffic / get_max_traffic(),
            ])
        return obs

    def get_gnn_enhanced_state(self):
        """Enhanced observation with GNN embeddings"""
        # Get node features
        node_features = self.graph.get_node_features(self.deploymentList)
        
        # Apply GNN if enabled
        with torch.no_grad():
            gnn_embeddings = self.gnn(node_features, self.graph.adjacency_matrix)
        
        # Combine original features with GNN embeddings
        obs = []
        for i in range(len(self.deploymentList)):
            # Original features (normalized)
            deployment = self.deploymentList[i]
            original_features = [
                deployment.num_pods / MAX_REPLICATION,
                deployment.desired_replicas / MAX_REPLICATION,
                deployment.cpu_usage / get_max_cpu(),
                deployment.mem_usage / get_max_mem(),
                deployment.received_traffic / get_max_traffic(),
                deployment.transmit_traffic / get_max_traffic(),
                deployment.latency / 1000.0 if hasattr(deployment, 'latency') else 0.0,
            ]
            
            # GNN embeddings
            gnn_features = gnn_embeddings[i].numpy().tolist()
            
            # Combine both
            obs.extend(original_features + gnn_features)
        
        return obs

    def calculate_reward(self):
        """Calculate reward based on cost and latency"""
        cost_reward = get_cost_reward(self.deploymentList)
        latency_reward = get_latency_reward_online_boutique(ID_recommendation, self.deploymentList)
        return [cost_reward, latency_reward]

    def get_pareto_front_metrics(self):
        """Return metrics needed for Pareto front analysis"""
        return {
            'cost_rewards': self.avg_cost_rewards,
            'latency_rewards': self.avg_latency_rewards,
            'total_cost': self.total_cost_reward,
            'total_latency': self.total_latency_reward
        }
        
    def scalarize_reward(self, rewards, weights=[0.5, 0.5]):
        """Convert multi-objective rewards to single scalar"""
        return weights[0] * rewards[0] + weights[1] * rewards[1]

    def simulation_update(self):
        """Update simulation data"""
        if len(self.df) == 0:
            # Use random data if no simulation data available
            for i, deployment in enumerate(self.deploymentList):
                deployment.cpu_usage = np.random.randint(50, 200)
                deployment.mem_usage = np.random.randint(100, 500)
                deployment.received_traffic = np.random.randint(100, 1000)
                deployment.transmit_traffic = np.random.randint(100, 1000)
                deployment.latency = np.random.uniform(0.1, 2.0)
        else:
            # Use real simulation data from CSV
            # Select a random row from the dataset to simulate dynamic behavior
            sample = self.df.sample(n=1).iloc[0]
            
            # Update each deployment with data from the selected row
            for i, deployment_name in enumerate(DEPLOYMENTS):
                deployment = self.deploymentList[i]
                
                # Extract metrics from the CSV row
                try:
                    deployment.num_pods = int(sample[f'{deployment_name}_num_pods'])
                    deployment.desired_replicas = int(sample[f'{deployment_name}_desired_replicas'])
                    deployment.cpu_usage = float(sample[f'{deployment_name}_cpu_usage'])
                    deployment.mem_usage = float(sample[f'{deployment_name}_mem_usage'])
                    deployment.received_traffic = float(sample[f'{deployment_name}_traffic_in'])
                    deployment.transmit_traffic = float(sample[f'{deployment_name}_traffic_out'])
                    deployment.latency = float(sample[f'{deployment_name}_latency'])
                except KeyError as e:
                    logging.warning(f"Column {e} not found in dataset, using default values")
                    # Fallback to random values if column is missing
                    deployment.cpu_usage = np.random.randint(50, 200)
                    deployment.mem_usage = np.random.randint(100, 500)
                    deployment.received_traffic = np.random.randint(100, 1000)
                    deployment.transmit_traffic = np.random.randint(100, 1000)
                    deployment.latency = np.random.uniform(0.1, 2.0)

        for d in self.deploymentList:
            d.update_replicas()
        
    def get_graph_info(self):
        """Return graph structure information for analysis"""
        return {
            'adjacency_matrix': self.graph.adjacency_matrix.numpy(),
            'service_names': self.graph.service_names,
            'num_services': self.graph.num_services
        }

    def visualize_graph(self):
        """Visualize the microservice dependency graph"""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            # Create NetworkX graph
            G = nx.from_numpy_array(self.graph.adjacency_matrix.numpy())
            
            # Add node labels
            labels = {i: name for i, name in enumerate(self.graph.service_names)}
            
            # Plot
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, k=2, iterations=50)
            nx.draw(G, pos, labels=labels, with_labels=True, 
                   node_color='lightblue', node_size=1500, 
                   font_size=8, font_weight='bold')
            plt.title("Online Boutique Microservice Dependency Graph")
            plt.tight_layout()
            plt.savefig("microservice_graph.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ Graph visualization saved as 'microservice_graph.png'")
            
        except ImportError:
            print("❌ matplotlib or networkx not available for visualization")

    def get_dataset_info(self):
        """Return information about the loaded dataset"""
        if len(self.df) == 0:
            return {
                'dataset_loaded': False,
                'message': 'No dataset loaded, using random simulation data'
            }
        else:
            return {
                'dataset_loaded': True,
                'dataset_shape': self.df.shape,
                'columns': list(self.df.columns),
                'date_range': {
                    'start': self.df['date'].min() if 'date' in self.df.columns else 'N/A',
                    'end': self.df['date'].max() if 'date' in self.df.columns else 'N/A'
                },
                'sample_count': len(self.df),
                'services_included': [col.split('_')[0] for col in self.df.columns if '_num_pods' in col]
            }
