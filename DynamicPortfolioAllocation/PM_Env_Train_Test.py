import numpy as np

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Train/Test helpers
def make_train_test_envs(
        PM_Environment,
        price_tensor,
        cost_tensor,
        #return_tensor,
        feature_tensor,
        feature_mask,
        tickers,
        lookback=30,
        train_ratio=0.8,
        initial_cash=100_000,
):
    T = feature_tensor.shape[0]
    split = int(T * train_ratio)

    assert price_tensor.shape[:2] == feature_tensor.shape[:2]
    assert cost_tensor.shape == feature_tensor.shape[:2]
    assert feature_mask.shape == feature_tensor.shape[:2]


    train_env = PM_Environment(
        price_tensor=price_tensor,
        cost_tensor = cost_tensor,
        #return_tensor=return_tensor,
        feature_tensor=feature_tensor,
        feature_mask=feature_mask,
        tickers=tickers,
        lookback=lookback,
        initial_cash=initial_cash,
        start_index=lookback,
        end_index=split,
    )

    test_env = PM_Environment(
        price_tensor=price_tensor,
        cost_tensor = cost_tensor,
        #return_tensor=return_tensor
        feature_tensor=feature_tensor,
        feature_mask=feature_mask,
        tickers=tickers,
        lookback=lookback,
        initial_cash=initial_cash,
        start_index=split + lookback, # to avoid training overlap
        end_index=T - 1,
    )

    return train_env, test_env