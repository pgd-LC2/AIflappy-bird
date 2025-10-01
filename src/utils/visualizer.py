"""
训练数据可视化工具
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import json
import os


def moving_average(data: List[float], window: int = 100) -> List[float]:
    """计算移动平均"""
    if len(data) < window:
        return data
    
    weights = np.ones(window) / window
    return np.convolve(data, weights, mode='valid')


def plot_training_curves(log_file: str, save_path: str = None):
    """
    绘制训练曲线
    
    参数:
        log_file: 日志文件路径
        save_path: 保存图片的路径（可选）
    """
    with open(log_file, 'r') as f:
        log_data = json.load(f)
    
    episode_scores = log_data['episode_scores']
    episode_rewards = log_data['episode_rewards']
    episode_lengths = log_data['episode_lengths']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    ax1 = axes[0]
    episodes = list(range(1, len(episode_scores) + 1))
    ax1.plot(episodes, episode_scores, alpha=0.3, label='原始得分', color='blue')
    
    if len(episode_scores) >= 100:
        smoothed_scores = moving_average(episode_scores, window=100)
        smoothed_episodes = list(range(50, 50 + len(smoothed_scores)))
        ax1.plot(smoothed_episodes, smoothed_scores, label='移动平均(100)', 
                color='red', linewidth=2)
    
    ax1.axhline(y=50, color='green', linestyle='--', linewidth=2, label='目标(50分)')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('得分', fontsize=12)
    ax1.set_title('训练得分曲线', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(episodes, episode_rewards, alpha=0.3, label='原始奖励', color='orange')
    
    if len(episode_rewards) >= 100:
        smoothed_rewards = moving_average(episode_rewards, window=100)
        smoothed_episodes = list(range(50, 50 + len(smoothed_rewards)))
        ax2.plot(smoothed_episodes, smoothed_rewards, label='移动平均(100)', 
                color='darkred', linewidth=2)
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('累计奖励', fontsize=12)
    ax2.set_title('训练奖励曲线', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    ax3.plot(episodes, episode_lengths, alpha=0.3, label='原始长度', color='purple')
    
    if len(episode_lengths) >= 100:
        smoothed_lengths = moving_average(episode_lengths, window=100)
        smoothed_episodes = list(range(50, 50 + len(smoothed_lengths)))
        ax3.plot(smoothed_episodes, smoothed_lengths, label='移动平均(100)', 
                color='darkviolet', linewidth=2)
    
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('步数', fontsize=12)
    ax3.set_title('Episode长度曲线', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    else:
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        print("训练曲线已保存到: training_curves.png")
    
    plt.close()


def plot_evaluation_results(scores: List[int], seeds: List[int], save_path: str = None):
    """
    绘制评估结果
    
    参数:
        scores: 各个种子的得分列表
        seeds: 随机种子列表
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.bar(range(len(scores)), scores, color='steelblue', alpha=0.7)
    ax1.axhline(y=np.mean(scores), color='red', linestyle='--', linewidth=2, 
                label=f'平均得分: {np.mean(scores):.2f}')
    ax1.axhline(y=50, color='green', linestyle='--', linewidth=2, label='目标(50分)')
    ax1.set_xlabel('测试轮次', fontsize=12)
    ax1.set_ylabel('得分', fontsize=12)
    ax1.set_title('不同随机种子下的表现', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.boxplot([scores], labels=['所有测试'])
    ax2.axhline(y=50, color='green', linestyle='--', linewidth=2, label='目标(50分)')
    ax2.set_ylabel('得分', fontsize=12)
    ax2.set_title('得分分布统计', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    stats_text = f'平均: {np.mean(scores):.2f}\n'
    stats_text += f'标准差: {np.std(scores):.2f}\n'
    stats_text += f'最高: {np.max(scores)}\n'
    stats_text += f'最低: {np.min(scores)}\n'
    stats_text += f'中位数: {np.median(scores):.2f}'
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"评估结果已保存到: {save_path}")
    else:
        plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
        print("评估结果已保存到: evaluation_results.png")
    
    plt.close()
