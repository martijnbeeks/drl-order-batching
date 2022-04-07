from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.cmd_util import make_vec_env
from drl_env import WAREHOUSE
import random
import numpy as np


class ObjectiveFunction:
    def __init__(self):
        self.version_number = 0
        self.weight_tardy_list = []
        self.weight_picking_list = []

    def objective(self, parameters):
        weight_tardy, weight_picking = parameters
        self.weight_tardy_list.append(weight_tardy)
        self.weight_picking_list.append(weight_picking)

        heuristic = "BOC"

        # Instantiate the env
        env = WAREHOUSE(params=parameters, heuristic=heuristic)
        env = make_vec_env(lambda: env, n_envs=1)

        # Retrain a base agent
        model = PPO2.load("./trained_models/batching_operation/benchmark/V001_second_run")
        model.set_env(env)
        model.gamma = 0.999999

        models = 2
        training_steps = 1000000
        results_all = []
        # wrap it
        for i in range(1, models):
            model.learn(total_timesteps=training_steps)

            env = WAREHOUSE(params=parameters, heuristic=heuristic)
            obs = env.reset()
            steps = 100000
            results_tardy_orders = []
            results_picking_time = []

            for step in range(steps):
                valid_actions = env.sim.available_actions(obs)
                valid_actions = [idx for idx, x in enumerate(valid_actions) if x > 0]

                action, _states = model.predict(obs)
                if action in valid_actions:
                    obs, rewards, done, info = env.step(action)
                else:
                    action = random.sample(valid_actions, 1)[0]
                    obs, rewards, done, info = env.step(action)

                if done:
                    results = env.sim.episode_render_test()
                    obs = env.reset()
                    results_tardy_orders.append(results['tardy_orders'])
                    results_picking_time.append(results['picking_time'])

            results_all.append(np.multiply(results_tardy_orders, results_picking_time))

        self.version_number += 1

        return np.mean(results_all[0])

