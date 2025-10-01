"""
PPO智能体实现
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
from .ppo_network import ActorCritic, RolloutBuffer


class PPOAgent:
    """PPO智能体"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化PPO智能体
        
        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            lr: 学习率
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
            clip_epsilon: PPO裁剪参数
            value_loss_coef: 价值损失系数
            entropy_coef: 熵损失系数
            max_grad_norm: 梯度裁剪阈值
            n_epochs: 每次更新的训练轮数
            batch_size: 批量大小
            device: 设备（cpu或cuda）
        """
        self.device = torch.device(device)
        
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self.buffer = RolloutBuffer()
        
        self.total_updates = 0
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        选择动作
        
        返回: (action, log_prob, state_value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, state_value = self.policy(state_tensor)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
                log_prob = torch.log(action_probs[0, action])
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), state_value.item()
    
    def store_transition(self, state, action, reward, state_value, log_prob, done):
        """存储转移"""
        self.buffer.add(state, action, reward, state_value, log_prob, done)
    
    def compute_gae(self, rewards: torch.Tensor, state_values: torch.Tensor, 
                    dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算广义优势估计（Generalized Advantage Estimation）
        
        返回: (advantages, returns)
        """
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = state_values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - state_values[t]
            
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + state_values
        
        return advantages, returns
    
    def update(self) -> dict:
        """
        使用收集的经验更新策略
        
        返回: 训练统计信息
        """
        states, actions, rewards, state_values, old_log_probs, dones = self.buffer.get()
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        state_values = state_values.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        dones = dones.to(self.device)
        
        with torch.no_grad():
            advantages, returns = self.compute_gae(rewards, state_values, dones)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        n_updates = 0
        
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                new_log_probs, new_state_values, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(new_state_values, batch_returns)
                
                entropy_loss = -entropy.mean()
                
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_updates += 1
        
        self.total_updates += 1
        
        self.buffer.clear()
        
        stats = {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy_loss': total_entropy_loss / n_updates,
            'total_loss': (total_policy_loss + total_value_loss + total_entropy_loss) / n_updates
        }
        
        return stats
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_updates': self.total_updates
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_updates = checkpoint.get('total_updates', 0)


import torch.nn.functional as F
