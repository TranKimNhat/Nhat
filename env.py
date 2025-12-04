"""
EV Charging Station Environment - Layer 2 RL Controller
========================================================
Version 10: FIX UNMET VÔ LÝ

VẤN ĐỀ V9:
==========
- Agent học cách GIẢM EV serving (action_ev < 0)
- Gây ra unmet vô lý dù tất cả giá đều có lãi
- Success rate = 0%

GIẢI PHÁP V10:
==============
1. VÔ HIỆU HÓA action_ev: Với giá mới (tất cả < 3858), L1 đã serve 100%
   → Agent không cần điều chỉnh EV
   → delta_ev = 0 luôn (ignore action[1])

2. TĂNG C_UNMET: 1000 → 10000 VND/kWh
   → Penalty rất cao nếu có unmet

3. TĂNG LAMBDA_TERMINAL: 2000 → 5000
   → Ép SoC về 500 mạnh hơn

GIÁ MỚI:
========
- Grid Buy: 1631, 2601, 3835 VND/kWh (tất cả < 3858)
- Grid Sell: 1012 VND/kWh
- EV Charge: 3858 VND/kWh
- → LUÔN CÓ LÃI khi serve EV!

Author: Claude
Date: 2024
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ChargingStationEnv(gym.Env):
    """
    EV Charging Station Environment for RL Layer 2.
    Version 10: Fixed irrational unmet by disabling action_ev.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, data: dict, num_steps: int = 96, lookahead_horizon: int = 8):
        super().__init__()
        
        self.data = data
        self.num_steps = num_steps
        self.current_step = 0
        
        # === SYSTEM PARAMETERS (IEEE-AM Paper) ===
        self.DELTA_T = 0.25
        self.PV_MAX_POWER_KW = 500.0
        self.EV_MAX_POWER_KW = 1000.0
        self.GRID_MAX_POWER_KW = 1000.0
        self.BESS_CAPACITY_KWH = 1000.0
        self.BESS_MAX_POWER_KW = 500.0
        self.BESS_ETA_CH = 0.95
        self.BESS_ETA_DCH = 0.95
        self.SOC_MIN_KWH = 200.0
        self.SOC_MAX_KWH = 1000.0
        
        # === PRICES (VND/kWh) ===
        self.PRICE_EV = 3858.0
        self.PRICE_SELL = 1012.0
        self.C_UNMET = 10000.0  # TĂNG TỪ 1000 → 10000
        
        # Price normalization
        self.PRICE_NORM = 4000.0
        self.PRICE_PEAK_THRESHOLD = 3500.0
        
        # === ACTION SCALING ===
        self.DELTA_BESS_MAX = 200.0  # kW
        self.DELTA_EV_MAX = 0.0      # VÔ HIỆU HÓA! (was 100.0)
        
        # === REWARD HYPERPARAMETERS (V10) ===
        self.REWARD_SCALE = 1e-5
        self.SOC_TARGET = 500.0
        
        # P_action: Light penalty
        self.LAMBDA_ACTION = 200.0
        
        # P_terminal: TĂNG từ 2000 → 5000
        self.LAMBDA_TERMINAL = 5000.0
        self.TERMINAL_START_STEP = 80
        
        # === ACTION SPACE ===
        # Vẫn giữ 2D để không thay đổi agent architecture
        # Nhưng action[1] sẽ bị ignore
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # === STATE SPACE ===
        self.LOOKAHEAD_HORIZON = lookahead_horizon
        self.state_dim = 5 + 6 + 7 + (5 * self.LOOKAHEAD_HORIZON)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        # === INTERNAL STATE ===
        self.current_soc_kwh = 500.0
        self.prev_soc_kwh = 500.0
        self.cumulative_profit = 0.0
        
        self._precompute_daily_stats()
    
    def _precompute_daily_stats(self):
        """Pre-compute daily statistics."""
        prices = self.data.get('grid_price', np.ones(self.num_steps) * 2601)
        self.avg_remaining_prices = np.zeros(self.num_steps)
        for t in range(self.num_steps):
            if t < self.num_steps - 1:
                self.avg_remaining_prices[t] = np.mean(prices[t+1:])
            else:
                self.avg_remaining_prices[t] = prices[t]
        
        self.peak_price = np.max(prices)
        self.off_peak_price = np.min(prices)
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        
        self.current_step = 0
        
        if 'l1_soc_schedule' in self.data and len(self.data['l1_soc_schedule']) > 0:
            self.current_soc_kwh = self.data['l1_soc_schedule'][0]
        else:
            self.current_soc_kwh = 500.0
        
        self.prev_soc_kwh = self.current_soc_kwh
        self.cumulative_profit = 0.0
        
        self._precompute_daily_stats()
        
        return self._get_state(), self._get_empty_info()
    
    def _get_state(self):
        """Construct state vector (58D)."""
        t = self.current_step
        
        pv_actual = self.data['pv_actual'][t]
        pv_forecast = self.data.get('pv_forecast', self.data['pv_actual'])[t]
        ev_demand = self.data['ev_demand_actual'][t]
        ev_forecast = self.data['ev_demand_forecast'][t]
        grid_price = self.data['grid_price'][t]
        
        l1_ev = self.data.get('l1_power_schedule', np.zeros(self.num_steps))[t]
        l1_bess = self.data.get('l1_bess_schedule', np.zeros(self.num_steps))[t]
        l1_soc = self.data.get('l1_soc_schedule', np.ones(self.num_steps) * 500)[t]
        l1_grid = self.data.get('l1_grid_schedule', np.zeros(self.num_steps))[t]
        
        # 1. REAL-TIME (5D)
        real_time = np.array([
            self.current_soc_kwh / self.BESS_CAPACITY_KWH,
            pv_actual / self.PV_MAX_POWER_KW,
            ev_demand / self.EV_MAX_POWER_KW,
            grid_price / self.PRICE_NORM,
            t / self.num_steps,
        ], dtype=np.float32)
        
        # 2. FORECAST & L1 (6D)
        forecast_l1 = np.array([
            (pv_actual - pv_forecast) / self.PV_MAX_POWER_KW,
            (ev_demand - ev_forecast) / self.EV_MAX_POWER_KW,
            (self.current_soc_kwh - l1_soc) / self.BESS_CAPACITY_KWH,
            l1_ev / self.EV_MAX_POWER_KW,
            l1_bess / self.BESS_MAX_POWER_KW,
            l1_grid / self.GRID_MAX_POWER_KW,
        ], dtype=np.float32)
        
        # 3. FUTURE SUMMARY (7D)
        pv_remain = np.sum(self.data['pv_actual'][t+1:]) * self.DELTA_T if t < self.num_steps - 1 else 0
        ev_remain = np.sum(self.data['ev_demand_actual'][t+1:]) * self.DELTA_T if t < self.num_steps - 1 else 0
        prices_remaining = self.data['grid_price'][t+1:] if t < self.num_steps - 1 else []
        n_peak = np.sum(np.array(prices_remaining) >= self.PRICE_PEAK_THRESHOLD) / 20 if len(prices_remaining) > 0 else 0
        
        future_summary = np.array([
            pv_remain / (self.PV_MAX_POWER_KW * 24),
            ev_remain / (self.EV_MAX_POWER_KW * 24),
            n_peak,
            self.SOC_TARGET / self.BESS_CAPACITY_KWH,
            self.avg_remaining_prices[t] / self.PRICE_NORM,
            self._get_time_to_price_change(t) / self.num_steps,
            self.data['grid_price'][min(t+1, self.num_steps-1)] / self.PRICE_NORM,
        ], dtype=np.float32)
        
        # 4. LOOKAHEAD (5*8=40D)
        lookahead = []
        for h in range(self.LOOKAHEAD_HORIZON):
            future_t = min(t + h + 1, self.num_steps - 1)
            lookahead.extend([
                self.data['pv_actual'][future_t] / self.PV_MAX_POWER_KW,
                self.data['ev_demand_actual'][future_t] / self.EV_MAX_POWER_KW,
                self.data['grid_price'][future_t] / self.PRICE_NORM,
                self.data.get('l1_bess_schedule', np.zeros(self.num_steps))[future_t] / self.BESS_MAX_POWER_KW,
                self.data.get('l1_soc_schedule', np.ones(self.num_steps) * 500)[future_t] / self.BESS_CAPACITY_KWH,
            ])
        
        return np.concatenate([real_time, forecast_l1, future_summary, np.array(lookahead, dtype=np.float32)])
    
    def _get_time_to_price_change(self, t):
        current_price = self.data['grid_price'][t]
        for i in range(t + 1, self.num_steps):
            if self.data['grid_price'][i] != current_price:
                return i - t
        return self.num_steps - t
    
    def step(self, action):
        """Execute one step."""
        t = self.current_step
        
        # === 1. PARSE ACTION ===
        delta_bess_ratio = float(np.clip(action[0], -1, 1))
        # ACTION_EV BỊ VÔ HIỆU HÓA - luôn = 0
        delta_ev_ratio = 0.0  # IGNORE action[1]!
        
        delta_bess = delta_bess_ratio * self.DELTA_BESS_MAX
        delta_ev = 0.0  # Luôn = 0
        
        # === 2. GET CURRENT DATA ===
        pv_actual = self.data['pv_actual'][t]
        ev_demand = self.data['ev_demand_actual'][t]
        grid_price = self.data['grid_price'][t]
        
        l1_ev = self.data.get('l1_power_schedule', np.zeros(self.num_steps))[t]
        l1_bess = self.data.get('l1_bess_schedule', np.zeros(self.num_steps))[t]
        l1_soc = self.data.get('l1_soc_schedule', np.ones(self.num_steps) * 500)[t]
        
        # === 3. EV: FOLLOW L1 100% ===
        # Với giá mới (tất cả < 3858), L1 serve 100% EV
        target_ev = l1_ev  # Không điều chỉnh!
        target_ev = np.clip(target_ev, 0, ev_demand)
        
        # === 4. BESS: Agent điều chỉnh ===
        target_bess = l1_bess + delta_bess
        
        if target_bess > 0:  # Discharge
            max_dch = (self.current_soc_kwh - self.SOC_MIN_KWH) * self.BESS_ETA_DCH / self.DELTA_T
            max_dch = min(max_dch, self.BESS_MAX_POWER_KW)
            p_bess_dch = np.clip(target_bess, 0, max_dch)
            p_bess_ch = 0
        else:  # Charge
            max_ch = (self.SOC_MAX_KWH - self.current_soc_kwh) / (self.BESS_ETA_CH * self.DELTA_T)
            max_ch = min(max_ch, self.BESS_MAX_POWER_KW)
            p_bess_ch = np.clip(-target_bess, 0, max_ch)
            p_bess_dch = 0
        
        p_bess_net = p_bess_dch - p_bess_ch
        
        # === 5. POWER BALANCE ===
        available = pv_actual + p_bess_dch - p_bess_ch
        
        if target_ev > available:
            p_grid_buy = np.clip(target_ev - available, 0, self.GRID_MAX_POWER_KW)
            p_grid_sell = 0
            ev_served = min(target_ev, available + p_grid_buy)
        else:
            p_grid_sell = np.clip(available - target_ev, 0, self.GRID_MAX_POWER_KW)
            p_grid_buy = 0
            ev_served = target_ev
        
        ev_served = np.clip(ev_served, 0, ev_demand)
        ev_unmet = ev_demand - ev_served
        
        # === 6. UPDATE SOC ===
        delta_soc = (self.BESS_ETA_CH * p_bess_ch - p_bess_dch / self.BESS_ETA_DCH) * self.DELTA_T
        self.prev_soc_kwh = self.current_soc_kwh
        self.current_soc_kwh = np.clip(self.current_soc_kwh + delta_soc, self.SOC_MIN_KWH, self.SOC_MAX_KWH)
        
        # === 7. REWARD (V10) ===
        # R_economic
        rev_ev = ev_served * self.PRICE_EV * self.DELTA_T
        rev_sell = p_grid_sell * self.PRICE_SELL * self.DELTA_T
        cost_buy = p_grid_buy * grid_price * self.DELTA_T
        penalty_unmet = ev_unmet * self.C_UNMET * self.DELTA_T  # C_UNMET = 10000
        
        R_economic = rev_ev + rev_sell - cost_buy - penalty_unmet
        self.cumulative_profit += R_economic
        
        # P_action (only BESS action counts since EV is disabled)
        P_action = self.LAMBDA_ACTION * (delta_bess_ratio ** 2)
        
        # P_terminal
        urgency = self._calc_urgency(t)
        soc_error = self.current_soc_kwh - self.SOC_TARGET
        P_terminal = self.LAMBDA_TERMINAL * (soc_error ** 2) * urgency
        
        reward_raw = R_economic - P_action - P_terminal
        reward = reward_raw * self.REWARD_SCALE
        
        # === 8. STEP ===
        self.current_step += 1
        terminated = self.current_step >= self.num_steps
        
        # === 9. INFO ===
        info = {
            'step': t,
            'time_hour': t * self.DELTA_T,
            'soc_kwh': self.current_soc_kwh,
            'soc_percent': self.current_soc_kwh / self.BESS_CAPACITY_KWH * 100,
            'pv_actual': pv_actual,
            'ev_demand': ev_demand,
            'ev_served': ev_served,
            'ev_unmet': ev_unmet,
            'p_grid_buy': p_grid_buy,
            'p_grid_sell': p_grid_sell,
            'p_grid_net': p_grid_buy - p_grid_sell,
            'p_bess_ch': p_bess_ch,
            'p_bess_dch': p_bess_dch,
            'p_bess_net': p_bess_net,
            'grid_price': grid_price,
            'l1_ev': l1_ev,
            'l1_bess': l1_bess,
            'l1_soc': l1_soc,
            'action_bess': delta_bess_ratio,
            'action_ev': delta_ev_ratio,  # Luôn = 0
            'delta_bess': delta_bess,
            'delta_ev': delta_ev,
            'R_economic': R_economic,
            'P_action': P_action,
            'P_terminal': P_terminal,
            'reward_raw': reward_raw,
            'reward': reward,
            'profit_step': R_economic,
            'cumulative_profit': self.cumulative_profit,
        }
        
        if terminated:
            info['final_soc'] = self.current_soc_kwh
            info['total_profit'] = self.cumulative_profit
            info['is_valid'] = abs(self.current_soc_kwh - self.SOC_TARGET) <= 25
        
        if not terminated:
            next_state = self._get_state()
        else:
            # Return last valid state for terminal step
            self.current_step = self.num_steps - 1
            next_state = self._get_state()
            self.current_step = self.num_steps
        
        return next_state, reward, terminated, False, info
    
    def _calc_urgency(self, t):
        if t < self.TERMINAL_START_STEP:
            return 0.0
        progress = (t - self.TERMINAL_START_STEP) / (self.num_steps - self.TERMINAL_START_STEP)
        return progress ** 2
    
    def _get_empty_info(self):
        return {'step': 0, 'soc_kwh': self.current_soc_kwh, 'profit_step': 0, 'cumulative_profit': 0}
    
    def render(self, mode='human'):
        print(f"Step {self.current_step}: SoC={self.current_soc_kwh:.1f} kWh")


# ==============================================================================
# REWARD FORMULA DOCUMENTATION
# ==============================================================================

REWARD_FORMULA = """
================================================================================
REWARD FUNCTION V10 - FIX UNMET VÔ LÝ
================================================================================

THAY ĐỔI TỪ V9:
---------------
1. VÔ HIỆU HÓA action_ev: delta_ev = 0 luôn
   - Với giá mới (tất cả < 3858), L1 đã serve 100% EV
   - Agent chỉ cần quản lý BESS

2. TĂNG C_UNMET: 1000 → 10000 VND/kWh
   - Penalty cực cao nếu có unmet

3. TĂNG LAMBDA_TERMINAL: 2000 → 5000
   - Ép SoC về 500 mạnh hơn

CÔNG THỨC:
----------
R_total = R_economic - P_action - P_terminal

1. R_economic = Rev_EV + Rev_Sell - Cost_Buy - Penalty_Unmet
   - Penalty_Unmet = EV_unmet × 10000 × 0.25 (rất cao!)

2. P_action = 200 × action_bess²
   - Chỉ tính BESS action (EV disabled)

3. P_terminal = 5000 × (SoC - 500)² × α(t)
   - α(t) = ((t - 72) / 24)² if t ≥ 72

HYPERPARAMETERS:
----------------
| Parameter        | V9    | V10    |
|------------------|-------|--------|
| DELTA_EV_MAX     | 100   | **0**  |
| C_UNMET          | 1000  | **10000** |
| LAMBDA_TERMINAL  | 2000  | **5000** |
================================================================================
"""


if __name__ == '__main__':
    print(REWARD_FORMULA)