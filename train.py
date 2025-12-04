"""
SAC Training Script for EV Charging Station Layer 2
====================================================
Version 9: Updated for new prices (1631, 2601, 3835)

Uses:
- SAC Agent (agent.py)
- Env V9 (env_v9.py → rename to env.py)
- Training data: train_generated.xlsx

Author: Claude
Date: 2024
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import custom modules
from env import ChargingStationEnv, REWARD_FORMULA
from agent import SACAgent

# Try to import export_results, but make it optional
try:
    from export_results import export_episode_results, export_training_summary
    HAS_EXPORT = True
except ImportError:
    HAS_EXPORT = False
    print("Warning: export_results module not found, using basic export")


# ==============================================================================
# 1. DATA LOADING
# ==============================================================================

def load_data_file(filename):
    """Load and preprocess data file."""
    column_map = {
        # Actual data
        'PV (kW)': 'pv_actual',
        'EV demand actual (kW)': 'ev_demand_actual',
        
        # Forecasts
        'PV forecast (kW)': 'pv_forecast',
        'PV Sunny forecast (kW)': 'pv_forecast',
        'EV demand forecast (kW)': 'ev_demand_forecast',
        
        # L1 Solution
        'EV served (kW)': 'l1_power_schedule',
        'P_BESS_net (kW)': 'l1_bess_schedule',
        'P_grid_net (kW)': 'l1_grid_schedule',
        'SoC (kWh)': 'l1_soc_schedule',
        
        # Prices
        'Price_grid_buy (VND/kWh)': 'grid_price',
        'Price_grid_sell (VND/kWh)': 'sell_price',
        'Price_EV_charge (VND/kWh)': 'ev_price',
    }
    
    try:
        df = pd.read_excel(filename)
        print(f"Loaded {filename}: {len(df)} rows")
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return [], 0
    
    df_renamed = df.rename(columns=column_map)
    
    # Fill missing columns
    if 'pv_forecast' not in df_renamed.columns:
        df_renamed['pv_forecast'] = df_renamed.get('pv_actual', 0)
    if 'l1_grid_schedule' not in df_renamed.columns:
        df_renamed['l1_grid_schedule'] = 0.0
    
    num_steps = 96
    num_days = len(df_renamed) // num_steps
    
    day_data_list = []
    for day_idx in range(num_days):
        start = day_idx * num_steps
        end = start + num_steps
        
        day_data = {}
        for col in ['pv_actual', 'pv_forecast', 'ev_demand_actual', 'ev_demand_forecast',
                    'grid_price', 'l1_power_schedule', 'l1_bess_schedule', 
                    'l1_soc_schedule', 'l1_grid_schedule']:
            if col in df_renamed.columns:
                day_data[col] = df_renamed[col].iloc[start:end].values.astype(np.float32)
        
        day_data_list.append(day_data)
    
    print(f"  Split into {num_days} days")
    return day_data_list, num_days


# ==============================================================================
# 2. TRAINING FUNCTIONS
# ==============================================================================

def train_episode(env, agent, day_data):
    """Train one episode."""
    env.data = day_data
    if hasattr(env, '_precompute_daily_stats'):
        env._precompute_daily_stats()
    
    state, _ = env.reset()
    episode_reward = 0
    episode_profit = 0
    update_info = None
    done = False
    
    while not done:
        action = agent.select_action(state, deterministic=False)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        agent.store_transition(state, action, reward, next_state, done)
        update_info = agent.update()
        
        episode_reward += reward
        episode_profit += info.get('profit_step', 0)
        state = next_state
    
    return {
        'reward': episode_reward,
        'profit': episode_profit,
        'final_soc': env.current_soc_kwh,
        'is_valid': abs(env.current_soc_kwh - 500) <= 25,
        'update_info': update_info,
    }


def evaluate(env, agent, data_list, num_episodes=None):
    """Evaluate agent."""
    if num_episodes is None:
        num_episodes = len(data_list)
    
    profits, socs, valid = [], [], 0
    
    for i in range(min(num_episodes, len(data_list))):
        env.data = data_list[i]
        if hasattr(env, '_precompute_daily_stats'):
            env._precompute_daily_stats()
        
        state, _ = env.reset()
        ep_profit = 0
        done = False
        
        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_profit += info.get('profit_step', 0)
            state = next_state
        
        profits.append(ep_profit)
        socs.append(env.current_soc_kwh)
        if abs(env.current_soc_kwh - 500) <= 25:
            valid += 1
    
    return {
        'avg_profit': np.mean(profits),
        'avg_final_soc': np.mean(socs),
        'success_rate': valid / len(profits) if profits else 0,
    }


def run_detailed_episode(env, agent, day_data):
    """Run episode with detailed logging."""
    env.data = day_data
    if hasattr(env, '_precompute_daily_stats'):
        env._precompute_daily_stats()
    
    state, _ = env.reset()
    records = []
    done = False
    
    while not done:
        action = agent.select_action(state, deterministic=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        step = info['step']
        records.append({
            'Time (h)': step * 0.25,
            'PV_actual (kW)': info.get('pv_actual', 0),
            'EV demand_actual (kW)': info.get('ev_demand', 0),
            'EV served (kW)': info.get('ev_served', 0),
            'EV unmet (kW)': info.get('ev_unmet', 0),
            'P_BESS_net (kW)': info.get('p_bess_net', 0),
            'P_BESS_ch (kW)': info.get('p_bess_ch', 0),
            'P_BESS_dch (kW)': info.get('p_bess_dch', 0),
            'P_grid_buy (kW)': info.get('p_grid_buy', 0),
            'P_grid_sell (kW)': info.get('p_grid_sell', 0),
            'P_grid_net (kW)': info.get('p_grid_net', 0),
            'SoC (kWh)': info.get('soc_kwh', 500),
            'Price_grid_buy (VND/kWh)': info.get('grid_price', 0),
            'Price_grid_sell (VND/kWh)': info.get('sell_price', 1012),
            'Price_EV_charge (VND/kWh)': info.get('ev_price', 3858),
            'R_economic': info.get('R_economic', 0),
            'P_action': info.get('P_action', 0),
            'P_terminal': info.get('P_terminal', 0),
            'Reward': reward,
            'Action_BESS': info.get('action_bess', 0),
            'Action_EV': info.get('action_ev', 0),
        })
        state = next_state
    
    return pd.DataFrame(records)


# ==============================================================================
# 3. MAIN TRAINING LOOP
# ==============================================================================

def train_sac(
    train_file='train_generated.xlsx',
    test_file='input_test_1day.xlsx',
    num_epochs=5,
    save_dir='sac_results',
):
    """Main training function."""
    
    print("=" * 70)
    print("SAC TRAINING FOR EV CHARGING STATION (V9)")
    print("=" * 70)
    print(f"\nPrices: 1631, 2601, 3835 VND/kWh")
    print(f"All prices < EV price (3858) → Always profitable to serve!")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    print("\n📂 Loading data...")
    train_data, num_train_days = load_data_file(train_file)
    test_data, num_test_days = load_data_file(test_file)
    
    if not train_data:
        print("Error: No training data loaded!")
        return
    
    # Create environment and agent
    print("\n🔧 Creating environment and agent...")
    env = ChargingStationEnv(train_data[0])
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_entropy=True,
        buffer_capacity=100000,
        batch_size=256,
    )
    
    # Training loop
    print("\n🚀 Starting training...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Training days: {num_train_days}")
    
    history = {
        'epoch': [],
        'episode': [],
        'train_profit': [],
        'train_soc': [],
        'val_profit': [],
        'val_soc': [],
        'success_rate': [],
    }
    
    total_episodes = 0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'='*70}")
        
        # Shuffle training data
        indices = np.random.permutation(num_train_days)
        
        for i, day_idx in enumerate(indices):
            total_episodes += 1
            
            # Train episode
            result = train_episode(env, agent, train_data[day_idx])
            
            # Evaluate periodically
            if (i + 1) % 10 == 0 or i == 0:
                val_data = test_data if test_data else train_data[:5]
                val_result = evaluate(env, agent, val_data, num_episodes=min(5, len(val_data)))
                
                print(f"\nEp {total_episodes}: "
                      f"Train P={result['profit']/1e6:.2f}M SoC={result['final_soc']:.0f} | "
                      f"Val P={val_result['avg_profit']/1e6:.2f}M SR={val_result['success_rate']*100:.0f}%")
                
                history['epoch'].append(epoch + 1)
                history['episode'].append(total_episodes)
                history['train_profit'].append(result['profit'])
                history['train_soc'].append(result['final_soc'])
                history['val_profit'].append(val_result['avg_profit'])
                history['val_soc'].append(val_result['avg_final_soc'])
                history['success_rate'].append(val_result['success_rate'])
        
        # Save checkpoint
        agent.save(os.path.join(save_dir, f'sac_epoch_{epoch+1}.pt'))
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    if test_data:
        final_result = evaluate(env, agent, test_data)
        print(f"\nTest Results:")
        print(f"  Avg Profit: {final_result['avg_profit']/1e6:.2f}M VND")
        print(f"  Avg Final SoC: {final_result['avg_final_soc']:.1f} kWh")
        print(f"  Success Rate: {final_result['success_rate']*100:.1f}%")
    
    # Save final model
    agent.save(os.path.join(save_dir, 'sac_final.pt'))
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)
    
    # Run detailed episode and save results
    if test_data:
        results_df = run_detailed_episode(env, agent, test_data[0])
        results_df.to_excel(os.path.join(save_dir, 'SAC_Results_Summary.xlsx'), index=False)
        print(f"\n✅ Results saved to {save_dir}/SAC_Results_Summary.xlsx")
    
    print(f"\n✅ Training complete! Models saved to {save_dir}/")
    
    return agent, history


# ==============================================================================
# 4. ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train_generated.xlsx', help='Training data file')
    parser.add_argument('--test', default='input_test_1day.xlsx', help='Test data file')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--save-dir', default='sac_results', help='Save directory')
    
    args = parser.parse_args()
    
    train_sac(
        train_file=args.train,
        test_file=args.test,
        num_epochs=args.epochs,
        save_dir=args.save_dir,
    )