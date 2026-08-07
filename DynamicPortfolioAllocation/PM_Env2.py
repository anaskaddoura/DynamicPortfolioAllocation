# Amendments:
# Separated buy and sell turnover in the step function

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv


"""Define a main Portfolio Management Gymnasium (Gym) environment object.
# Typical Gym architecture: 
# 1. Initialization + action_space + observation_space
# 2. Constructing Observations
# 3. Reset Function
# 4. Step Function

# Additions:
# A.1. Process Action to ensure proper data types, values, and apply transformations if needed
# A.2. Render function to print metrics for debugging"""

class PM_Environment2(gym.Env):
    
    #1 initialization
    def __init__(
        self,
        # The environnment's attributes inspired by the methodology of Jiang et al. (2017) are: price_tensor, cost_tensor, feature_tensor, feature_mask, tickers, lookback, initial_cash, log_ret_c_index (number od matrices of log returns on cost)
        # start_index (index to start with test data), end_index (index to end with test_data)
        price_tensor: np.ndarray,
        cost_tensor: np.ndarray,
        feature_tensor: np.ndarray,
        feature_mask: np.ndarray,
        #return_tensor: np.ndarray,

        tickers: list[str] | None = None,
        lookback: int = 30,
        initial_cash: float = 100_000,
        #transaction_cost: float = 0.001,

        log_ret_c_index: int = 3,
        start_index: int | None = None,
        end_index: int | None = None,

    ):
        super().__init__()

    #1.1. Arguments preprocessing

    # 1.1.1. Force data types to be np arrays
        self.price_tensor = np.asarray(price_tensor, dtype=np.float32)
        self.cost_tensor_full = np.asarray(cost_tensor, dtype=np.float32)
        #self.return_tensor_full = np.asarray(return_tensor, dtype=np.float32)    
        self.feature_tensor_full = np.asarray(feature_tensor, dtype=np.float32)
        self.feature_mask_full = np.asarray(feature_mask, dtype=bool)
    

    #1.1.2. Ensure dimension compatibility, availability of tickers
        if self.feature_tensor_full.ndim != 3:
            raise ValueError("feature_tensor must have shape (T, N, F)")
        if self.feature_mask_full.ndim !=2:
            raise ValueError("feature_mask must have shape (T, N)")
    
        
    #1.1.3. Extract tensor dimensions and check for correct environment index boundaries
            # T: date-time index, N: number of assets, F: number of features (LogRetO, LogRetH...etc.)
        self.T_full, self.N, self.F = self.feature_tensor_full.shape

        if self.feature_mask_full.shape != (self.T_full, self.N):
            raise ValueError("feature_mask shape must match feature_tensor first two dimensions")
        
        if self.cost_tensor_full.shape != (self.T_full, self.N):
            raise ValueError("cost_tensor shape must match feature_tensor first two dimensions")
    
        #if self.return_tensor_full.shape != (self.T_full, self.N):
        #    raise ValueError("return_tensor must have shape (T, N)")
        
        # Extra robustness to keep track of assets
        self.tickers = tickers if tickers is not None else [f"Asset_{i}" for i in range(self.N)]
    
        if len(self.tickers) != self.N:
            raise ValueError("len(tickers) must equal number of assets N")
            

        # The lookback is the number of periods that the agent observes and learns from
        self.lookback = lookback
        self.initial_cash = float(initial_cash)
        #self.transaction_cost = float(transaction_cost)
        self.log_ret_c_index = int(log_ret_c_index)
        
        if self.log_ret_c_index < 0 or self.log_ret_c_index >= self.F:
            raise ValueError("log_ret_c_index is outside feature dimension")
    
        self.start_index = lookback if start_index is None else max(start_index, lookback)
        # Tensor indexing is zero-based, so end_index is self.T_full - 1
        self.end_index = self.T_full - 1 if end_index is None else min(end_index, self.T_full - 1)
    
        if self.start_index >= self.end_index:
            raise ValueError("start_index must be smaller than end_index")



    #1.2. Define what actions are available (action space)
        # Action space = portfolio weights for N assets + cash.
        # The env will clip and normalize them
        # So the action space is a Box, low=0 (only chash), high=1 (1 asset only in portfolio)

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.N + 1,), # Weights for N assets + cash
            dtype=np.float32,
        )
    

    
    #1.3. Define what observations tha agent can observe (observation space)
    # The observation can have any value in Real space, it is the Log Ret of OHLC + LogVol Change of managed assets
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback * self.N * self.F,), # The observation is for lookback period for N assets, and F features
            dtype=np.float32,
        )
    
        self.t = None
        self.portfolio_value = None
        self.weights = None
        self.history = None

    
    #2 Constructing Observation = rolling window of market features (used by the agent model for learning and evaluation)
    def _get_obs(self):
        # The observation is the full feature tensor at t-lookback
        # We have already set the start_index to the minimum of lookback (line 97)
        obs = self.feature_tensor_full[self.t - self.lookback:self.t]
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        # The faeture tensor is flattened
        return obs.astype(np.float32).flatten()


    #3 Reset Function: used by the agent model for learning and evaluation
    # This resets:
    #   Time to start_index
    #   portfolio_value to initial_cash
    #   Assets' weights to 0, cash weight to 1
    #   last_drifted_weights to be the same as reset weights
    #   history to an empty list
    #   obs to an observation at t = start_index

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)

        self.t = self.start_index
        self.portfolio_value = self.initial_cash

        # N stock weights + 1 cash weight.
        self.weights = np.zeros(self.N + 1, dtype=np.float32)
        self.weights[-1] = 1.0

        self.last_drifted_weights = self.weights.copy()

        self.history = []

        obs = self._get_obs()
        # We store info for tracking
        info = {
            "portfolio_value": self.portfolio_value,
            "weights": self.weights.copy(),
        }

        return obs, info
    
    #4 Step Function: used by the agent model for learning and evaluation
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        
        old_weights = self.weights.copy()

       # Price relatives y_t: for cash = 1 (USD=USD), assets = exp(log return)

        log_ret_c = self.feature_tensor_full[self.t, :, self.log_ret_c_index]

        asset_price_relatives = np.exp(log_ret_c)
        asset_price_relatives = np.nan_to_num(
            asset_price_relatives,
            nan=1.0,
            posinf=1.0,
            neginf=1.0
        )

        y = np.append(asset_price_relatives, 1.0).astype(np.float32)

        # 4.1. Portfolio grows before rebalancing
        gross_growth = float(np.dot(old_weights, y))


        # Before moving forward, apply a complete portfolio loss filter
        # i.e. ensure that the portfolio still has value to be managed
        if gross_growth <= 0:
            reward = -10.0 # Extremely penalize reward in case of complete portfolio value loss
            terminated = True # And terminate the episode since there are no more assets to manage
            # Without this constraint, the portfolio value might go to negative 
            # Which is not within the scope of our environment
            truncated = False # Truncated is related to interruptions external to environment dynamics
            obs = self._get_obs()
            return obs, reward, terminated, truncated, {}

                
        # 4.2. Weight drift after price movement
        # Calculate the adjusted/factored weights of each asset after portfolio growth
        # It is important to calculate drifted weights to calculate accurate transaction costs (check section 4.3 below)
        drifted_weights = (old_weights * y) / gross_growth


        # Target portfolio weights after rebalancing
        # _process_action is applied to thhe action to ensure:
        #   Proper data type and value domain
        #   Tradability
        # and apply weight smoothing/regularization if necessary
        target_weights = self._process_action(action)

        # 4.3. Transaction remainder factor mu, inspired by Jiang et al. (2017)
        # This is not the exact implementation of Jiang et al. (2017)'s transaction remainder factor
        # Howver it solves for a transaction remainder factor that is a function of transaction costs
        # Where transaction costs are a function of the transaction remainder factor
        # Hence the iterative solution methodwith convergin criteria

        # Start by extracting costs from the modeled cost tensor
        cost_rates_assets = self.cost_tensor_full[self.t]
        cost_rates_assets = np.nan_to_num(
            cost_rates_assets,
            nan=0.0005,
            posinf=0.02,
            neginf=0.0005
        )

        # Cash has zero transaction cost since USD=USD (no forex operations)
        cost_rates = np.append(cost_rates_assets, 0.0)

        # Compute the remainder factor iteratively
        # Mu is a function of buy and sell cost rates
        # At the same time, buy and sell fractions are a function of Mu through the adjusted_target = mu*target_weights
        # Start with zero costs, mu=1.0
        mu = 1.0
        for _ in range(20):
            adjusted_target = mu * target_weights
            
            # Compute buy weights
            buy = np.clip(adjusted_target - drifted_weights, 0, None)
            # Compute sell weights
            sell = np.clip(drifted_weights - adjusted_target, 0, None)

            # Transaction remainder factor is what is left after accounting for transaction cost
            mu_next = 1.0 - float(np.sum(cost_rates * (buy + sell)))

            # Converging criteria: this is when adjusted targets are almost equal to target_weights
            if abs(mu_next - mu) < 1e-10:
                break

            mu = mu_next

        mu = float(np.clip(mu, 0.0, 1.0))

        # 4.4. Net portfolio growth: discount transaction remainder factor
        net_growth = mu * gross_growth
        net_return = net_growth - 1.0

        old_value = self.portfolio_value
        self.portfolio_value *= net_growth

        # Check for complete loss of portfolio value
        if self.portfolio_value <= 0:
            reward = -10.0
            terminated = True
        else:
            reward = float(np.log(net_growth))
            terminated = False

        # 4.5. After rebalancing, portfolio weights equal target_weights
        self.weights = target_weights

        # Tracked Diagnostics for later evaluation
        delta_from_drift = target_weights - drifted_weights
        buy_turnover = float(np.sum(np.clip(delta_from_drift[:-1], 0, None)))
        sell_turnover = float(np.sum(np.clip(-delta_from_drift[:-1], 0, None)))
        turnover = buy_turnover + sell_turnover
        transaction_cost = gross_growth * (1.0 - mu)

        
        # Store post-market drifted state for PVM approximation
        # PVM approximation is done in a child class

        self.last_drifted_weights = drifted_weights.copy()
       
        
        # Time step
        self.t += 1
        # Check termination condition
        if self.t >= self.end_index:
            terminated = True

        truncated = False

        
        obs = self._get_obs()

        info = {
            "portfolio_value": self.portfolio_value,
            "gross_growth": gross_growth,
            "net_growth": net_growth,
            "net_return": net_return,
            "transaction_remainder_factor": mu,
            "transaction_cost": transaction_cost,
            "turnover": turnover,
            "buy_turnover": buy_turnover,
            "sell_turnover": sell_turnover,
            "weights": self.weights.copy(),
            "drifted_weights": drifted_weights.copy(),
            "target_weights": target_weights.copy(),
            "cost_rates": cost_rates_assets.copy(),
            "t": self.t,
        }

        self.history.append(info)

        return obs, reward, terminated, truncated, info
    
    
    # Additional functions/helpers


    #A1. _process_action
    # Ensure proper data type and value domain
    # Ensure tradability
    # Apply weight smoothing/regularization if necessary
    def _process_action(self, action):
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action = np.clip(action, 0.0, 1.0)

        if action.sum() <= 0:
            action = np.zeros(self.N + 1, dtype=np.float32)
            action[-1] = 1.0
        else:
            action = action / action.sum()

        stock_weights = action[:-1]
        cash_weight = action[-1]

        # Apply tradability mask
        tradable = self.feature_mask_full[self.t]
        stock_weights = stock_weights * tradable.astype(np.float32)

        max_weight = 0.25
        stock_weights = np.minimum(stock_weights, max_weight)

        total = stock_weights.sum() + cash_weight

        if total <= 0:
            stock_weights = np.zeros(self.N, dtype=np.float32)
            cash_weight = 1.0
        else:
            stock_weights = stock_weights / total
            cash_weight = cash_weight / total

        # Weight smoothing: moving 10% towards target every day to avoid overtrading
        # This shall lower turnover (preventing pathological reallocations 
        # and mimicking realistic institutional execution)
        # Simple way for stable policy implementation without destabalizing model's hyperparameters
        
        target_weights = np.append(stock_weights, cash_weight)

        #rebalance_strength = 0.10

        #new_weights = (
        #    (1 - rebalance_strength) * self.weights
        #    + rebalance_strength * target_weights # Only move 10% towards target
        #)
        #return new_weights.astype(np.float32)

        # Instead of: return  np.append(stock_weights, cash_weight).astype(np.float32)

        return target_weights.astype(np.float32)

    
    #A.2. render function could be used for debugging
    # This was added to track portfolio value when cost drag was very high
    def render(self):
        print(f"t={self.t}, value={self.portfolio_value:,.2f}")





class PM_Environment2_PVM(PM_Environment2): # The Porfolio Vector Memory environment inspired by Jiang et al. (2017) is constructed as a child class to the main environment parent class/object
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Add one feature channel for previous stock weight
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.lookback * self.N * (self.F + 1),),
                dtype=np.float32,
            )

        

        def _get_obs(self):
            obs = self.feature_tensor_full[self.t - self.lookback:self.t]

            obs = np.nan_to_num(
                obs,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            # Previous stock weights only, excluding cash, post market after drift
            prev_stock_weights = self.last_drifted_weights[:-1]

            # Repeat across lookback window
            weight_channel = np.tile(
                prev_stock_weights.reshape(1, self.N, 1),
                (self.lookback, 1, 1)
            )

            obs_pvm = np.concatenate(
                [obs, weight_channel],
                axis=-1
            )

            return obs_pvm.astype(np.float32).flatten()
        





##########################################################################################################################


class PM_Environment2_s(gym.Env):
    #1 initialization
    def __init__(
        self,
        # The environnment's attributes inspired by the methodology of Jiang et al. (2017) are: price_tensor, cost_tensor, feature_tensor, feature_mask, tickers, lookback, initial_cash, log_ret_c_index (number od matrices of log returns on cost)
        # start_index (index to start with test data), end_index (index to end with test_data)
        price_tensor: np.ndarray,
        cost_tensor: np.ndarray,
        feature_tensor: np.ndarray,
        feature_mask: np.ndarray,
        #return_tensor: np.ndarray,

        tickers: list[str] | None = None,
        lookback: int = 30,
        initial_cash: float = 100_000,
        #transaction_cost: float = 0.001,

        log_ret_c_index: int = 3,
        start_index: int | None = None,
        end_index: int | None = None,

    ):
        super().__init__()

        self.price_tensor = np.asarray(price_tensor, dtype=np.float32)
        self.cost_tensor_full = np.asarray(cost_tensor, dtype=np.float32)
        #self.return_tensor_full = np.asarray(return_tensor, dtype=np.float32)    

        self.feature_tensor_full = np.asarray(feature_tensor, dtype=np.float32)
        self.feature_mask_full = np.asarray(feature_mask, dtype=bool)
    
        # Ensure dimension compatibility, availability of tickers, 

        if self.feature_tensor_full.ndim != 3:
            raise ValueError("feature_tensor must have shape (T, N, F)")
        if self.feature_mask_full.ndim !=2:
            raise ValueError("feature_mask must have shape (T, N)")
    
        self.T_full, self.N, self.F = self.feature_tensor_full.shape

        if self.feature_mask_full.shape != (self.T_full, self.N):
            raise ValueError("feature_mask shape must match feature_tensor first two dimensions")
        
        if self.cost_tensor_full.shape != (self.T_full, self.N):
            raise ValueError("cost_tensor shape must match feature_tensor first two dimensions")
    
        #if self.return_tensor_full.shape != (self.T_full, self.N):
        #    raise ValueError("return_tensor must have shape (T, N)")
        
        self.tickers = tickers if tickers is not None else [f"Asset_{i}" for i in range(self.N)]
    
        if len(self.tickers) != self.N:
            raise ValueError("len(tickers) must equal number of assets N")
            

        self.lookback = lookback
        self.initial_cash = float(initial_cash)
        #self.transaction_cost = float(transaction_cost)
        self.log_ret_c_index = int(log_ret_c_index)
    
        if self.log_ret_c_index >= self.F:
            raise ValueError("log_ret_c_index is outside feature dimension")
        
        if self.log_ret_c_index < 0 or self.log_ret_c_index >= self.F:
            raise ValueError("log_ret_c_index is outside feature dimension")
    
        self.start_index = lookback if start_index is None else max(start_index, lookback)
        self.end_index = self.T_full - 1 if end_index is None else min(end_index, self.T_full - 1)
    
        if self.start_index >= self.end_index:
            raise ValueError("start_index must be smaller than end_index")



        # Action = portfolio weights for N assets + cash.
        # The env will clip and normalize them.
        # So the action space is a Box, low=0, high=1 (1 asset only in portfolio)

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.N + 1,), # N assets + cash
            dtype=np.float32,
        )
    

    #2 Constructing Observation = rolling window of market features.
    # The observation can have any value in Real space, it is the Log Ret of OHLC of managed assets
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback * self.N * self.F,),
            dtype=np.float32,
        )
    
        self.t = None
        self.portfolio_value = None
        self.weights = None
        self.history = None

            
    #3 Reset Function    
    def reset(self, seed = None, options = None):
        super().reset(seed=seed)

        self.t = self.start_index
        self.portfolio_value = self.initial_cash

        # N stock weights + 1 cash weight.
        self.weights = np.zeros(self.N + 1, dtype=np.float32)
        self.weights[-1] = 1.0

        self.last_drifted_weights = self.weights.copy()

        self.history = []

        obs = self._get_obs()
        # We store info for tracking
        info = {
            "portfolio_value": self.portfolio_value,
            "weights": self.weights.copy(),
        }

        return obs, info
    
    #4 Step Function
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        
        old_weights = self.weights.copy()

       # Price relatives y_t: cash = 1, risky assets = exp(log return)

        log_ret_c = self.feature_tensor_full[self.t, :, self.log_ret_c_index]
        asset_price_relatives = np.exp(log_ret_c)

        #simple_returns = self.return_tensor_full[self.t]

        #simple_returns = np.nan_to_num(
        #    simple_returns,
        #    nan=0.0,
        #   posinf=0.0,
        #   neginf=0.0
        #)

        #asset_price_relatives = 1.0 + simple_returns

        asset_price_relatives = np.nan_to_num(
            asset_price_relatives,
            nan=1.0,
            posinf=1.0,
            neginf=1.0
        )

        y = np.append(asset_price_relatives, 1.0).astype(np.float32)

        # 1. Portfolio grows before rebalancing
        gross_growth = float(np.dot(old_weights, y))

        if gross_growth <= 0:
            reward = -10.0
            terminated = True
            truncated = False
            obs = self._get_obs()
            return obs, reward, terminated, truncated, {}

        # 2. Weight drift after price movement
        drifted_weights = (old_weights * y) / gross_growth


        # Target portfolio weights after rebalancing
        target_weights = self._process_action(action)

        # 3. Jiang et al.-style transaction remainder factor mu
        cost_rates_assets = self.cost_tensor_full[self.t]
        cost_rates_assets = np.nan_to_num(
            cost_rates_assets,
            nan=0.0005,
            posinf=0.02,
            neginf=0.0005
        )

        # cash has zero transaction cost
        cost_rates = np.append(cost_rates_assets, 0.0)

        mu = 1.0

        for _ in range(20):
            adjusted_target = mu * target_weights

            buy = np.clip(adjusted_target - drifted_weights, 0, None)
            sell = np.clip(drifted_weights - adjusted_target, 0, None)

            mu_next = 1.0 - float(np.sum(cost_rates * (buy + sell)))

            if abs(mu_next - mu) < 1e-10:
                break

            mu = mu_next

        mu = float(np.clip(mu, 0.0, 1.0))

        # 4. Net portfolio growth
        net_growth = mu * gross_growth
        net_return = net_growth - 1.0

        old_value = self.portfolio_value
        self.portfolio_value *= net_growth

        if self.portfolio_value <= 0:
            reward = -10.0
            terminated = True
        else:
            reward = float(np.log(net_growth))
            terminated = False

        # 5. After rebalancing, portfolio weights equal target_weights
        self.weights = target_weights

        # Diagnostics
        delta_from_drift = target_weights - drifted_weights
        buy_turnover = float(np.sum(np.clip(delta_from_drift[:-1], 0, None)))
        sell_turnover = float(np.sum(np.clip(-delta_from_drift[:-1], 0, None)))
        turnover = buy_turnover + sell_turnover
        transaction_cost = gross_growth * (1.0 - mu)

        
        # Store post-market drifted state for PVM approximation

        self.last_drifted_weights = drifted_weights.copy()
       
        self.t += 1

        if self.t >= self.end_index:
            terminated = True

        truncated = False

        
        obs = self._get_obs()

        info = {
            "portfolio_value": self.portfolio_value,
            "gross_growth": gross_growth,
            "net_growth": net_growth,
            "net_return": net_return,
            "transaction_remainder_factor": mu,
            "transaction_cost": transaction_cost,
            "turnover": turnover,
            "buy_turnover": buy_turnover,
            "sell_turnover": sell_turnover,
            "weights": self.weights.copy(),
            "drifted_weights": drifted_weights.copy(),
            "target_weights": target_weights.copy(),
            "cost_rates": cost_rates_assets.copy(),
            "t": self.t,
        }

        self.history.append(info)

        return obs, reward, terminated, truncated, info
    
    def _process_action(self, action):
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action = np.clip(action, 0.0, 1.0)

        if action.sum() <= 0:
            action = np.zeros(self.N + 1, dtype=np.float32)
            action[-1] = 1.0
        else:
            action = action / action.sum()

        stock_weights = action[:-1]
        cash_weight = action[-1]

        # Apply tradability mask
        tradable = self.feature_mask_full[self.t]
        stock_weights = stock_weights * tradable.astype(np.float32)

        max_weight = 0.25
        stock_weights = np.minimum(stock_weights, max_weight)

        total = stock_weights.sum() + cash_weight

        if total <= 0:
            stock_weights = np.zeros(self.N, dtype=np.float32)
            cash_weight = 1.0
        else:
            stock_weights = stock_weights / total
            cash_weight = cash_weight / total

        # Weight smoothing: moving 10% towards target every day to avoid overtrading
        # This shall lower turnover (preventing pathological reallocations 
        # and mimicking realistic institutional execution)
        
        target_weights = np.append(stock_weights, cash_weight)

        rebalance_strength = 0.10

        new_weights = (
            (1 - rebalance_strength) * self.weights
            + rebalance_strength * target_weights # Only move 10% towards target
        )
        return new_weights.astype(np.float32)

        # Instead of: return  np.append(stock_weights, cash_weight).astype(np.float32)

        #return target_weights.astype(np.float32)

    def _get_obs(self):
        obs = self.feature_tensor_full[self.t - self.lookback:self.t]
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs.astype(np.float32).flatten()
    
    def render(self):
        print(f"t={self.t}, value={self.portfolio_value:,.2f}")





class PM_Environment2_PVM_s(PM_Environment2_s): # The Porfolio Vector Memory environment inspired by Jiang et al. (2017) is constructed as a child class to the main environment parent class/object
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Add one feature channel for previous stock weight
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.lookback * self.N * (self.F + 1),),
                dtype=np.float32,
            )

        

        def _get_obs(self):
            obs = self.feature_tensor_full[self.t - self.lookback:self.t]

            obs = np.nan_to_num(
                obs,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            # Previous stock weights only, excluding cash, post market after drift
            prev_stock_weights = self.last_drifted_weights[:-1]

            # Repeat across lookback window
            weight_channel = np.tile(
                prev_stock_weights.reshape(1, self.N, 1),
                (self.lookback, 1, 1)
            )

            obs_pvm = np.concatenate(
                [obs, weight_channel],
                axis=-1
            )

            return obs_pvm.astype(np.float32).flatten()




