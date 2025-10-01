"""
训练日志记录器
"""
import json
import os
from datetime import datetime
from typing import Dict, List


class Logger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str = 'logs'):
        """
        初始化日志记录器
        
        参数:
            log_dir: 日志保存目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'training_{timestamp}.json')
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scores = []
        self.training_stats = []
        
        self.current_episode = 0
        self.total_steps = 0
    
    def log_episode(self, episode: int, reward: float, length: int, score: int, 
                   extra_info: Dict = None):
        """
        记录一个episode的信息
        
        参数:
            episode: episode编号
            reward: 累计奖励
            length: episode长度
            score: 游戏得分
            extra_info: 额外信息
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_scores.append(score)
        
        self.current_episode = episode
        self.total_steps += length
        
        print(f"Episode {episode}: Score={score}, Reward={reward:.2f}, Length={length}")
        
        if extra_info:
            for key, value in extra_info.items():
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    def log_training_stats(self, stats: Dict):
        """记录训练统计信息"""
        self.training_stats.append(stats)
    
    def get_recent_performance(self, n: int = 100) -> Dict:
        """
        获取最近n个episode的平均性能
        
        返回:
            包含平均奖励、平均得分、平均长度的字典
        """
        if len(self.episode_rewards) == 0:
            return {'avg_reward': 0, 'avg_score': 0, 'avg_length': 0}
        
        n = min(n, len(self.episode_rewards))
        
        return {
            'avg_reward': sum(self.episode_rewards[-n:]) / n,
            'avg_score': sum(self.episode_scores[-n:]) / n,
            'avg_length': sum(self.episode_lengths[-n:]) / n,
            'max_score': max(self.episode_scores[-n:]),
            'min_score': min(self.episode_scores[-n:])
        }
    
    def save(self):
        """保存日志到文件"""
        log_data = {
            'total_episodes': self.current_episode,
            'total_steps': self.total_steps,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_scores': self.episode_scores,
            'training_stats': self.training_stats,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\n日志已保存到: {self.log_file}")
    
    def print_summary(self):
        """打印训练摘要"""
        if len(self.episode_rewards) == 0:
            print("没有训练数据")
            return
        
        recent = self.get_recent_performance(100)
        all_time = self.get_recent_performance(len(self.episode_rewards))
        
        print("\n" + "="*60)
        print("训练摘要")
        print("="*60)
        print(f"总Episode数: {self.current_episode}")
        print(f"总步数: {self.total_steps}")
        print(f"\n最近100个Episode:")
        print(f"  平均得分: {recent['avg_score']:.2f}")
        print(f"  平均奖励: {recent['avg_reward']:.2f}")
        print(f"  平均长度: {recent['avg_length']:.2f}")
        print(f"  最高得分: {recent['max_score']}")
        print(f"\n全部Episode:")
        print(f"  平均得分: {all_time['avg_score']:.2f}")
        print(f"  最高得分: {all_time['max_score']}")
        print(f"  最低得分: {all_time['min_score']}")
        print("="*60 + "\n")
