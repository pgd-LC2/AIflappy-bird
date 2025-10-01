"""
PPO算法的Actor-Critic神经网络实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class ActorCritic(nn.Module):
    """Actor-Critic网络结构"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super(ActorCritic, self).__init__()
        
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """前向传播"""
        features = self.shared_layer(state)
        
        action_logits = self.actor(features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        state_value = self.critic(features)
        
        return action_probs, state_value
    
    def get_action(self, state, deterministic=False):
        """
        根据状态选择动作
        deterministic: 如果为True，选择概率最大的动作；否则根据概率分布采样
        """
        with torch.no_grad():
            action_probs, state_value = self.forward(state)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                dist = Categorical(action_probs)
                action = dist.sample()
            
            return action.item(), action_probs, state_value
    
    def evaluate_actions(self, states, actions):
        """
        评估给定状态和动作
        返回：动作的对数概率、状态价值、策略熵
        """
        action_probs, state_values = self.forward(states)
        
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return action_log_probs, state_values.squeeze(-1), entropy


class RolloutBuffer:
    """经验回放缓冲区，用于存储训练数据"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.state_values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, state, action, reward, state_value, log_prob, done):
        """添加一条经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def get(self):
        """获取所有数据并转换为tensor"""
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(np.array(self.actions))
        rewards = torch.FloatTensor(np.array(self.rewards))
        state_values = torch.FloatTensor(np.array(self.state_values))
        log_probs = torch.FloatTensor(np.array(self.log_probs))
        dones = torch.FloatTensor(np.array(self.dones))
        
        return states, actions, rewards, state_values, log_probs, dones
    
    def __len__(self):
        return len(self.states)
