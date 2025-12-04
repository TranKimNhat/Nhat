"""
Soft Actor-Critic (SAC) Agent for EV Charging Station
======================================================
Version 1.0

SAC advantages over DDPG:
- Stochastic policy with automatic exploration (entropy bonus)
- Twin Q-networks to reduce overestimation bias
- More stable and less sensitive to hyperparameters
- Better sample efficiency

Author: Claude
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import os


# ==============================================================================
# NEURAL NETWORK ARCHITECTURES
# ==============================================================================

class GaussianActor(nn.Module):
    """
    Stochastic Actor that outputs mean and log_std of Gaussian distribution.
    Actions are sampled and squashed through tanh.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], 
                 log_std_min=-20, log_std_max=2):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared layers
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # Small weights for output layers
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
    
    def forward(self, state):
        """Forward pass returning mean and log_std."""
        x = self.shared(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state):
        """
        Sample action from the policy.
        Returns: action, log_prob
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Sample with gradient
        
        # Squash through tanh
        action = torch.tanh(x_t)
        
        # Compute log probability with correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound (SAC paper appendix C)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state, deterministic=False):
        """Get action for evaluation."""
        mean, log_std = self.forward(state)
        
        if deterministic:
            return torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            return torch.tanh(x_t)


class TwinQNetwork(nn.Module):
    """
    Twin Q-Networks (Q1 and Q2) to reduce overestimation bias.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        
        # Q1 Network
        q1_layers = []
        prev_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            q1_layers.append(nn.Linear(prev_dim, hidden_dim))
            q1_layers.append(nn.LayerNorm(hidden_dim))
            q1_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        q1_layers.append(nn.Linear(hidden_dims[-1], 1))
        self.q1 = nn.Sequential(*q1_layers)
        
        # Q2 Network (same architecture, different weights)
        q2_layers = []
        prev_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            q2_layers.append(nn.Linear(prev_dim, hidden_dim))
            q2_layers.append(nn.LayerNorm(hidden_dim))
            q2_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        q2_layers.append(nn.Linear(hidden_dims[-1], 1))
        self.q2 = nn.Sequential(*q2_layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action):
        """Forward pass for both Q networks."""
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)
    
    def q1_forward(self, state, action):
        """Forward pass for Q1 only (used in actor update)."""
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)


# ==============================================================================
# REPLAY BUFFER
# ==============================================================================

class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


# ==============================================================================
# SAC AGENT
# ==============================================================================

class SACAgent:
    """
    Soft Actor-Critic Agent.
    
    Key features:
    - Stochastic policy with entropy regularization
    - Twin Q-networks
    - Automatic entropy tuning
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=[256, 256],
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_entropy=True,
        target_entropy=None,
        buffer_capacity=100000,
        batch_size=256,
        device=None
    ):
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"SAC Agent using device: {self.device}")
        
        # Dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_entropy = auto_entropy
        
        # Networks
        self.actor = GaussianActor(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic = TwinQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = TwinQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        
        # Copy weights to target
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Entropy tuning
        if auto_entropy:
            # Target entropy = -dim(A)
            if target_entropy is None:
                self.target_entropy = -action_dim
            else:
                self.target_entropy = target_entropy
            
            # Learnable log_alpha
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Training stats
        self.train_steps = 0
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []
        self.alpha_values = []
    
    def select_action(self, state, deterministic=False, add_noise=False):
        """
        Select action given state.
        add_noise parameter kept for compatibility but ignored (SAC explores via entropy)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor.get_action(state_tensor, deterministic=deterministic)
            return action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self):
        """
        Perform one update step.
        Returns: dict with losses
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # === UPDATE CRITIC ===
        with torch.no_grad():
            # Sample next actions and compute their log probs
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # Compute target Q values
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Current Q values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # === UPDATE ACTOR ===
        # Sample actions for current states
        new_actions, log_probs = self.actor.sample(states)
        
        # Q value of new actions
        q1_new = self.critic.q1_forward(states, new_actions)
        
        # Actor loss: minimize -Q + alpha * log_prob
        actor_loss = (self.alpha * log_probs - q1_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # === UPDATE ALPHA (if auto) ===
        alpha_loss = None
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # === SOFT UPDATE TARGET ===
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Track stats
        self.train_steps += 1
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        if alpha_loss is not None:
            self.alpha_losses.append(alpha_loss.item())
        self.alpha_values.append(self.alpha)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha_loss': alpha_loss.item() if alpha_loss else 0,
            'alpha': self.alpha,
            'q_value': current_q1.mean().item(),
        }
    
    def save(self, path):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'train_steps': self.train_steps,
        }
        
        if self.auto_entropy:
            checkpoint['log_alpha'] = self.log_alpha.detach().cpu()
            checkpoint['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.train_steps = checkpoint.get('train_steps', 0)
        
        if self.auto_entropy and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha'].to(self.device).requires_grad_(True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
            if 'alpha_optimizer' in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            self.alpha = self.log_alpha.exp().item()
        
        print(f"Model loaded from {path}")


# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing SAC Agent")
    print("=" * 60)
    
    # Create agent
    state_dim = 58
    action_dim = 2
    
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        auto_entropy=True,
    )
    
    print(f"\nActor parameters: {sum(p.numel() for p in agent.actor.parameters()):,}")
    print(f"Critic parameters: {sum(p.numel() for p in agent.critic.parameters()):,}")
    
    # Test action selection
    state = np.random.randn(state_dim).astype(np.float32)
    action = agent.select_action(state)
    print(f"\nTest action: {action}")
    print(f"Action in range [-1, 1]: {np.all(np.abs(action) <= 1)}")
    
    # Test training step
    print("\nFilling buffer with random experiences...")
    for _ in range(500):
        s = np.random.randn(state_dim).astype(np.float32)
        a = np.random.uniform(-1, 1, action_dim).astype(np.float32)
        r = np.random.randn()
        ns = np.random.randn(state_dim).astype(np.float32)
        d = False
        agent.store_transition(s, a, r, ns, d)
    
    print(f"Buffer size: {len(agent.replay_buffer)}")
    
    # Test update
    for i in range(10):
        result = agent.update()
        if result and i % 5 == 0:
            print(f"Update {i}: actor_loss={result['actor_loss']:.4f}, "
                  f"critic_loss={result['critic_loss']:.4f}, alpha={result['alpha']:.4f}")
    
    # Test save/load
    agent.save('test_sac_model.pt')
    agent.load('test_sac_model.pt')
    os.remove('test_sac_model.pt')
    
    print("\n✅ All tests passed!")