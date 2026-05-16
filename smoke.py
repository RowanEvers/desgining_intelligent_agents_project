from environments.wrappers import make_env
from environments.utils import Logger
from agents.reactive.sac import SACagent
log_dir = "logs/sac_smoke_test"

env = make_env("Swimmer-v5", seed=0)
logger = Logger(agent_name='SAC',env_id='Swimmer-v5',seed=0,log_dir=log_dir)
agent = SACagent(env=env, seed=0, log_dir=log_dir, logger=logger, device="gpu", learning_rate=3e-4)
agent.train(total_timesteps=10000)
agent.save(f"{log_dir}/final.zip")
logger.close()