from environments.wrappers import make_env
from agents.reactive.sac import SACagent

env = make_env("Swimmer-v5", seed=0, render_mode="human")
agent = SACagent.load(
    "logs/sac_smoke_test/SAC_Swimmer-v5_seed0/final",
    env=env, seed=0, log_dir=".", logger=None, device="cpu"
)


for ep in range(3):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Episode {ep}: return = {total_reward:.1f}")

env.close()