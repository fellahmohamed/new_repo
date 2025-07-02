import csv
import datetime
from datetime import datetime
import logging
import time
from statistics import mean

# MORL-Baselines requires the environment to be a mo-gym environment
import mo_gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

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


# MODIFIED: Inherit from mo_gym.MOEnv for multi-objective reinforcement learning
class OnlineBoutiqueMOV2(mo_gym.MOEnv):
    """
    Horizontal Scaling for Online Boutique in Kubernetes - a mo-gym environment for MORL.
    This environment is designed to be used with the morl-baselines library.
    It returns a vector reward for two objectives: cost and latency.
    """

    metadata = {'render.modes': ['human', 'ansi', 'array']}

    # MODIFIED: Removed goal_reward and objective_weights from __init__
    # These are now handled by the MORL agent/wrapper, not the environment.
    def __init__(self, k8s=False, waiting_period=0.3):
        # Define action and observation space
        # They must be gym.spaces objects

        super(OnlineBoutiqueMOV2, self).__init__()

        self.k8s = k8s
        self.name = "online_boutique_gym"
        self.__version__ = "0.0.1"
        self.seed()
        self.waiting_period = waiting_period  # seconds to wait after action
        
        # MODIFIED: Added reward_dim and reward_space for mo-gym compatibility
        self.reward_dim = 2  # Two objectives: cost and latency
        self.reward_space = spaces.Box(low=np.array([-np.inf, -np.inf]),
                                       high=np.array([np.inf, np.inf]),
                                       dtype=np.float32)

        logging.info("[Init] Env: {} | K8s: {} | Version {} | MO-Gym Enabled".format(self.name, self.k8s, self.__version__))

        # Current Step
        self.current_step = 0

        # Actions identified by integers 0-n -> 15 actions!
        self.num_actions = 15

        # Multi-Discrete Action Space
        # [Deployment_ID, Action_ID]
        self.action_space = spaces.MultiDiscrete([11, self.num_actions])

        # Observations: 6 metrics per deployment * 11 deployments = 66 metrics
        self.min_pods = MIN_REPLICATION
        self.max_pods = MAX_REPLICATION
        self.num_apps = 11 # Corrected from 2 to 11

        # Deployment Data
        self.deploymentList = get_online_boutique_deployment_list(self.k8s, self.min_pods, self.max_pods)

        # Logging Deployment
        for d in self.deploymentList:
            d.print_deployment()

        self.observation_space = self.get_observation_space()

        # Info
        self.avg_pods = []
        self.avg_latency = []

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
        self.df = pd.read_csv("../../datasets/real/" + self.deploymentList[0].namespace + "/v1/"
                              + self.name + '_' + 'observation.csv')

        # Trackers for multi-objective rewards
        self.total_cost_reward = 0
        self.total_latency_reward = 0
        self.avg_cost_rewards = []
        self.avg_latency_rewards = []
        
    def step(self, action):
        if self.current_step == 1:
            if not self.k8s:
                self.simulation_update()

            self.time_start = time.time()

        # Map action index to deployment ID
        n = action[ID_DEPLOYMENTS]

        # Execute one time step within the environment
        self.take_action(action[ID_MOVES], n)

        # Wait a few seconds if on real k8s cluster
        if self.k8s:
            if action[ID_MOVES] != ACTION_DO_NOTHING \
                    and self.constraint_min_pod_replicas is False \
                    and self.constraint_max_pod_replicas is False:
                time.sleep(self.waiting_period)

        # Update observation before reward calculation
        if self.k8s:
            for d in self.deploymentList:
                d.update_obs_k8s()
        else:
            self.simulation_update()

        # MODIFIED: Get the multi-objective reward vector
        reward_vector = self.get_reward
        self.total_cost_reward += reward_vector[0]
        self.total_latency_reward += reward_vector[1]

        self.avg_cost_rewards.append(reward_vector[0])
        self.avg_latency_rewards.append(reward_vector[1])
        
        # MODIFIED: Removed scalarization. The environment returns the raw reward vector.
        # The MORL agent is responsible for handling the vector.

        self.avg_pods.append(get_num_pods(self.deploymentList))
        self.avg_latency.append(self.deploymentList[0].latency) # Note: tracks latency of first service

        logging.info('[Step {}] | Action (Deployment): {} | Action (Move): {} | Reward Vector: {} | Total Cost Reward: {} | Total Latency Reward: {}'.format(
            self.current_step, DEPLOYMENTS[action[0]], MOVES[action[1]], reward_vector, self.total_cost_reward, self.total_latency_reward))

        ob = self.get_state()
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_obs_to_csv(self.obs_csv, np.array(ob), date, self.deploymentList[0].latency)

        self.info = dict(
            total_cost_reward=self.total_cost_reward,
            total_latency_reward=self.total_latency_reward,
            cost_reward=reward_vector[0],
            latency_reward=reward_vector[1]
        )

        # Update Reward Keywords
        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False

        if self.current_step == MAX_STEPS:
            self.episode_count += 1
            self.execution_time = time.time() - self.time_start
            # Example of saving aggregate results at episode end
            # save_to_csv(self.file_results, self.episode_count, mean(self.avg_pods), mean(self.avg_latency),
            #             f"[{self.total_cost_reward}, {self.total_latency_reward}]", self.execution_time)

        # MODIFIED: Return the observation, the multi-objective reward vector, done flag, and info
        return np.array(ob), reward_vector, self.episode_over, self.info    

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
        # Render the environment to the screen (optional)
        pass

    def take_action(self, action, id):
        self.current_step += 1

        if self.current_step >= MAX_STEPS:
            self.episode_over = True
        
        # Action mapping logic...
        if action == ACTION_DO_NOTHING:
            pass
        elif ACTION_ADD_1_REPLICA <= action <= ACTION_ADD_7_REPLICA:
            num_replicas_to_add = action
            self.deploymentList[id].deploy_pod_replicas(num_replicas_to_add, self)
        elif ACTION_TERMINATE_1_REPLICA <= action <= ACTION_TERMINATE_7_REPLICA:
            num_replicas_to_terminate = action - (ACTION_TERMINATE_1_REPLICA - 1)
            self.deploymentList[id].terminate_pod_replicas(num_replicas_to_terminate, self)
        else:
            logging.info('[Take Action] Unrecognized Action: ' + str(action))

    @property
    def get_reward(self):
        """ 
        Calculate Rewards. 
        MODIFIED: Returns a numpy array for the multi-objective reward vector [cost, latency]. 
        """
        if self.constraint_max_pod_replicas or self.constraint_min_pod_replicas:
            # Return a penalty vector for violating constraints
            return np.array([-1.0, -3000.0], dtype=np.float32)
        
        reward_vector = self.calculate_reward()
        return reward_vector

    def get_state(self):
        # Observations: 6 metrics * 11 deployments = 66 metrics
        # "num_pods", "desired_replicas", "cpu_usage", "mem_usage", "received_traffic", "transmit_traffic"
        
        ob_list = []
        for d in self.deploymentList:
            ob_list.extend([
                d.num_pods,
                d.desired_replicas,
                d.cpu_usage,
                d.mem_usage,
                d.received_traffic,
                d.transmit_traffic
            ])
        return np.array(ob_list)

    def get_observation_space(self):
        # Create low and high bounds for all 11 services
        low_bounds = []
        high_bounds = []
        for _ in range(self.num_apps):
            low_bounds.extend([
                self.min_pods,       # Number of Pods
                self.min_pods,       # Desired Replicas
                0,                   # CPU Usage (in m)
                0,                   # MEM Usage (in MiB)
                0,                   # Received traffic
                0                    # Transmit traffic
            ])
            high_bounds.extend([
                self.max_pods,       # Number of Pods
                self.max_pods,       # Desired Replicas
                get_max_cpu(),       # CPU Usage (in m)
                get_max_mem(),       # MEM Usage (in MiB)
                get_max_traffic(),   # Received traffic
                get_max_traffic()    # Transmit traffic
            ])

        return spaces.Box(
            low=np.array(low_bounds),
            high=np.array(high_bounds),
            dtype=np.float32
        )

    def calculate_reward(self):
        """
        Calculates the multi-objective reward.
        MODIFIED: Returns the reward as a numpy array.
        """
        cost_reward = get_cost_reward(self.deploymentList)
        latency_reward = get_latency_reward_online_boutique(ID_recommendation, self.deploymentList)
        
        return np.array([cost_reward, latency_reward], dtype=np.float32)

    def get_pareto_front_metrics(self):
        """Return metrics needed for Pareto front analysis"""
        return {
            'cost_rewards': self.avg_cost_rewards,
            'latency_rewards': self.avg_latency_rewards,
            'total_cost': self.total_cost_reward,
            'total_latency': self.total_latency_reward
        }
        
    # REMOVED: The scalarize_reward function is no longer needed in the environment.
    # MORL agents handle the scalarization of the reward vector externally.

    def simulation_update(self):
        if self.current_step == 1:
            # Get a random sample!
            sample = self.df.sample()
            for i in range(len(DEPLOYMENTS)):
                self.deploymentList[i].num_pods = int(sample[DEPLOYMENTS[i] + '_num_pods'].values[0])
                self.deploymentList[i].num_previous_pods = int(sample[DEPLOYMENTS[i] + '_num_pods'].values[0])
        else:
            pods = [d.num_pods for d in self.deploymentList]
            previous_pods = [d.num_previous_pods for d in self.deploymentList]
            diff = [p - pp for p, pp in zip(pods, previous_pods)]
            
            for i in range(len(DEPLOYMENTS)):
                self.df['diff-' + DEPLOYMENTS[i]] = self.df[DEPLOYMENTS[i] + '_num_pods'].diff()

            # Filter dataframe to find a matching state transition
            data = self.df
            for i in range(len(DEPLOYMENTS)):
                # This logic could be improved for better state matching
                filtered_data = data.loc[(data[DEPLOYMENTS[i] + '_num_pods'] == pods[i]) & (data['diff-' + DEPLOYMENTS[i]] == diff[i])]
                if not filtered_data.empty:
                    data = filtered_data
                else: # Fallback if no exact match is found
                    fallback_data = self.df.loc[self.df[DEPLOYMENTS[i] + '_num_pods'] == pods[i]]
                    if not fallback_data.empty:
                        data = fallback_data
                        
            sample = data.sample(n=1) if not data.empty else self.df.sample(n=1)

        for i in range(len(DEPLOYMENTS)):
            self.deploymentList[i].cpu_usage = int(sample[DEPLOYMENTS[i] + '_cpu_usage'].values[0])
            self.deploymentList[i].mem_usage = int(sample[DEPLOYMENTS[i] + '_mem_usage'].values[0])
            self.deploymentList[i].received_traffic = int(sample[DEPLOYMENTS[i] + '_traffic_in'].values[0])
            self.deploymentList[i].transmit_traffic = int(sample[DEPLOYMENTS[i] + '_traffic_out'].values[0])
            self.deploymentList[i].latency = float("{:.3f}".format(sample[DEPLOYMENTS[i] + '_latency'].values[0]))

        for d in self.deploymentList:
            d.update_replicas()

    def save_obs_to_csv(self, obs_file, obs, date, latency):
        # This function writes the current observation to a CSV file.
        # It's simplified here for brevity but the original logic is maintained.
        try:
            with open(obs_file, 'a+', newline='') as file:
                fields = ['date']
                for d in self.deploymentList:
                    fields.extend([f'{d.name}_num_pods', f'{d.name}_desired_replicas', f'{d.name}_cpu_usage',
                                   f'{d.name}_mem_usage', f'{d.name}_traffic_in', f'{d.name}_traffic_out', f'{d.name}_latency'])
                
                writer = csv.writer(file)
                # Write header if file is new/empty
                if file.tell() == 0:
                    writer.writerow(fields)

                row_data = [date]
                obs_idx = 0
                for d in self.deploymentList:
                    row_data.extend([
                        int(obs[obs_idx]),      # num_pods
                        int(obs[obs_idx+1]),  # desired_replicas
                        int(obs[obs_idx+2]),  # cpu_usage
                        int(obs[obs_idx+3]),  # mem_usage
                        int(obs[obs_idx+4]),  # traffic_in
                        int(obs[obs_dict+5]),  # traffic_out
                        float("{:.3f}".format(d.latency)) # latency
                    ])
                    obs_idx += 6
                writer.writerow(row_data)

        except Exception as e:
            logging.error(f"Failed to write to CSV {obs_file}: {e}")