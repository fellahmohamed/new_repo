import logging
import argparse

from setuptools.command.alias import alias
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from sb3_contrib import RecurrentPPO

from gym_hpa.envs import Redis, OnlineBoutique , OnlineBoutiqueMO
from stable_baselines3.common.callbacks import CheckpointCallback

# Multi-Objective RL imports
try:
    from morl_baselines.single_policy.ser.mo_q_learning import MOQLearning
    MORL_AVAILABLE = True
    print("✅ MORL-Baselines available for Multi-Objective Q-Learning")
except ImportError:
    MORL_AVAILABLE = False
    print("❌ MORL-Baselines not available. Install with: pip install morl-baselines[all]")

# Logging
from policies.util.util import test_model

logging.basicConfig(filename='run.log', filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

parser = argparse.ArgumentParser(description='Run ILP!')
parser.add_argument('--alg', default='ppo', help='The algorithm: ["ppo", "recurrent_ppo", "a2c"]')
parser.add_argument('--k8s', default=False, action="store_true", help='K8s mode')
parser.add_argument('--use_case', default='redis', help='Apps: ["redis", "online_boutique"]')
parser.add_argument('--goal', default='cost', help='Reward Goal: ["cost", "latency" ,"cost_latency", "cost_latency_Q"]')

parser.add_argument('--training', default=False, action="store_true", help='Training mode')
parser.add_argument('--testing', default=False, action="store_true", help='Testing mode')
parser.add_argument('--loading', default=False, action="store_true", help='Loading mode')
parser.add_argument('--load_path', default='logs/model/test.zip', help='Loading path, ex: logs/model/test.zip')
parser.add_argument('--test_path', default='logs/model/test.zip', help='Testing path, ex: logs/model/test.zip')

parser.add_argument('--steps', default=500, help='The steps for saving.')
parser.add_argument('--total_steps', default=5000, help='The total number of steps.')

args = parser.parse_args()


def get_model(alg, env, tensorboard_log):
    model = 0
    if alg == 'ppo':
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, n_steps=500)
    elif alg == 'recurrent_ppo':
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
    elif alg == 'a2c':
        model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)  # , n_steps=steps
    elif alg == 'mo_q_learning':
        print ('Using Multi-Objective Q-Learning')
        if not MORL_AVAILABLE:
            raise ImportError("MORL-Baselines not available. Install with: pip install morl-baselines[all]")
        model = MOQLearning(
            env=env,
            learning_rate=0.01,
            gamma=0.95,
            epsilon=0.1,
            epsilon_decay=0.99,
            min_epsilon=0.01,
            scalarization_method='weighted_sum',
            verbose=1
        )
    else:
        logging.info('Invalid algorithm!')

    return model


def get_load_model(alg, tensorboard_log, load_path):
    if alg == 'ppo':
        return PPO.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log, n_steps=500)
    elif alg == 'recurrent_ppo':
        return RecurrentPPO.load(load_path, reset_num_timesteps=False, verbose=1,
                                 tensorboard_log=tensorboard_log)  # n_steps=steps
    elif alg == 'a2c':
        return A2C.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log)
    else:
        logging.info('Invalid algorithm!')


def get_env(use_case, k8s, goal):
    env = 0
    if use_case == 'redis':
        env = Redis(k8s=k8s, goal_reward=goal)
    elif use_case == 'online_boutique':
        env = OnlineBoutique(k8s=k8s, goal_reward=goal)
    elif use_case == 'online_boutique_mo':
        env = OnlineBoutiqueMO(k8s=k8s, goal_reward=goal, waiting_period=0.3, objective_weights=[0.5, 0.5])
    else:
        logging.error('Invalid use_case!')
        raise ValueError('Invalid use_case!')

    return env


def main():
    # Import and initialize Environment
    logging.info(args)

    alg = args.alg
    k8s = args.k8s
    use_case = args.use_case
    goal = args.goal

    loading = args.loading
    load_path = args.load_path
    training = args.training
    testing = args.testing
    test_path = args.test_path

    steps = int(args.steps)
    total_steps = int(args.total_steps)

    env = get_env(use_case, k8s, goal)

    scenario = ''
    if k8s:
        scenario = 'real'
    else:
        scenario = 'simulated'

    tensorboard_log = "../../results/" + use_case + "/" + scenario + "/" + goal + "/"

    name = alg + "_env_" + env.name + "_goal_" + goal + "_k8s_" + str(k8s) + "_totalSteps_" + str(total_steps)

    # callback
    checkpoint_callback = CheckpointCallback(save_freq=steps, save_path="logs/" + name, name_prefix=name)

    if training:
        # Set environment for MO Q-Learning to use multi-objective environment
        if alg == 'mo_q_learning' and goal == 'cost_latency_Q':
            # Force use of multi-objective environment
            env = OnlineBoutiqueMO(k8s=k8s, goal_reward='cost_latency', waiting_period=0.3, objective_weights=[0.5, 0.5])
            print(f"✅ Using Multi-Objective Environment for {alg} with goal {goal}")
        
        if loading:  # resume training
            model = get_load_model(alg, tensorboard_log, load_path)
            model.set_env(env)
            model.learn(total_timesteps=total_steps, tb_log_name=name + "_run", callback=checkpoint_callback)
        else:
            model = get_model(alg, env, tensorboard_log)
            if alg == 'mo_q_learning':
                # Multi-objective training
                model.train(total_timesteps=total_steps)
            else:
                # Standard RL training
                model.learn(total_timesteps=total_steps, tb_log_name=name + "_run", callback=checkpoint_callback)

        # Save model
        if hasattr(model, 'save'):
            model.save(name)
        else:
            print(f"⚠️ Model {alg} does not support save() method")

    if testing:
        model = get_load_model(alg, tensorboard_log, test_path)
        test_model(model, env, n_episodes=100, n_steps=110, smoothing_window=5, fig_name=name + "_test_reward.png")


if __name__ == "__main__":
    main()
