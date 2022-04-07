from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.cmd_util import make_vec_env
from drl_env import WAREHOUSE

# Set batching heuristic
heuristic = "BOC"

# Initiate 4 simulation environments in parallel
env = WAREHOUSE(params=(1, 1), heuristic=heuristic)
env = make_vec_env(lambda: env, n_envs=4)

path = "trained_models/scenario_2/"
model = PPO2(MlpPolicy, env, verbose=0)

# Mount initiated environment on the trained agent and set gamma value
model.set_env(env)
model.gamma = 0.999999

# Train agent with 4M steps each and save
version_number = 6
model.learn(total_timesteps=100000)
model.save(path+"V00"+str(version_number)+"_first_run")

