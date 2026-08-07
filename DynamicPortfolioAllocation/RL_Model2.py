

# Amendments to Model1 are changes in hyperparameters
# We change: learning_rate=5e-5, clip_range=0.05, ent_coef=0.0
# This makes PPO much less aggressive. 

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv



"""Below is the RL function based on SB3 PPO model (Proximal Policy Optimization).
The function trains a RL model  with the input train_env, total_timesteps, learning_rate, n_steps,
batch_size, gamma, gae_lambda, clip_range, and ent_coef"""
def agent_ppo2(
    train_env,
    total_timesteps=100_000,

    learning_rate=5e-5,
    n_steps=2048,
    batch_size=256,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.05,
    ent_coef=0.0,

    seed=None,
    device="cpu",
):
    # SB3 expects vectorized environments.
    vec_env = DummyVecEnv([lambda: train_env])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,

        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        seed=seed,
    )

    # Store hyperparameters
    model.run_hyperparams = {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "total_timesteps": total_timesteps,
        "seed":seed,
    }

    model.learn(total_timesteps=total_timesteps)

    return model



# Model evaluation
def evaluate_model2(model, env):
    obs, info = env.reset()
    done = False

    values = [info["portfolio_value"]]
    weights_history = []

    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = np.squeeze(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        values.append(info["portfolio_value"])
        weights_history.append(info["weights"])

    values = np.array(values)
    weights_history = np.array(weights_history)

    total_return = values[-1] / values[0] - 1.0
    running_max = np.maximum.accumulate(values)
    drawdown = values / running_max - 1.0
    max_drawdown = drawdown.min()

    results = {
        "final_value": values[-1],
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "equity_curve": values,
        "weights_history": weights_history,
        "history": env.history,
    }

    return results