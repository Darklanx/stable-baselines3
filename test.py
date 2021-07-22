from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.min_atar import make_min_atar_env
env = make_min_atar_env('breakout', 0)
# It will check your custom environment and output additional warnings if needed
check_env(env)