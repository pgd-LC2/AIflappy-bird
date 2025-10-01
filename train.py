"""
PPO智能体训练脚本
"""
import argparse
import os
import numpy as np
import torch
from src.game import FlappyBirdEnv
from src.agent import PPOAgent
from src.utils import Logger


def train(args):
    """训练PPO智能体"""
    
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    env = FlappyBirdEnv(render_mode=None, seed=args.seed)
    
    agent = PPOAgent(
        state_dim=env.observation_space_shape[0],
        action_dim=env.action_space_n,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=args.device
    )
    
    logger = Logger(log_dir=args.log_dir)
    
    print("="*60)
    print("开始训练PPO智能体")
    print("="*60)
    print(f"训练Episodes: {args.n_episodes}")
    print(f"更新间隔: {args.update_interval} 步")
    print(f"学习率: {args.lr}")
    print(f"设备: {args.device}")
    print(f"随机种子: {args.seed if args.seed is not None else '未设置'}")
    print("="*60 + "\n")
    
    best_avg_score = 0
    episode = 0
    
    for episode in range(1, args.n_episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, log_prob, state_value = agent.select_action(state)
            
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, state_value, log_prob, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if len(agent.buffer) >= args.update_interval:
                stats = agent.update()
                logger.log_training_stats(stats)
        
        logger.log_episode(episode, episode_reward, episode_length, info['score'])
        
        if episode % args.eval_interval == 0:
            recent_perf = logger.get_recent_performance(100)
            avg_score = recent_perf['avg_score']
            
            print(f"\n{'='*60}")
            print(f"Episode {episode} 评估")
            print(f"{'='*60}")
            print(f"最近100个Episode平均得分: {avg_score:.2f}")
            print(f"最近100个Episode最高得分: {recent_perf['max_score']}")
            print(f"最近100个Episode平均奖励: {recent_perf['avg_reward']:.2f}")
            print(f"{'='*60}\n")
            
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                model_path = os.path.join(args.model_dir, 'best_model.pth')
                agent.save(model_path)
                print(f"✓ 保存最佳模型 (平均得分: {avg_score:.2f}) 到 {model_path}\n")
            
            if episode % (args.eval_interval * 5) == 0:
                checkpoint_path = os.path.join(args.model_dir, f'checkpoint_ep{episode}.pth')
                agent.save(checkpoint_path)
                print(f"✓ 保存检查点到 {checkpoint_path}\n")
        
        if episode >= 100:
            recent_perf = logger.get_recent_performance(100)
            if recent_perf['avg_score'] >= args.target_score:
                print(f"\n{'='*60}")
                print(f"🎉 达到目标！")
                print(f"最近100个Episode平均得分: {recent_perf['avg_score']:.2f} >= {args.target_score}")
                print(f"{'='*60}\n")
                break
    
    final_model_path = os.path.join(args.model_dir, 'final_model.pth')
    agent.save(final_model_path)
    print(f"\n✓ 保存最终模型到 {final_model_path}")
    
    logger.save()
    logger.print_summary()
    
    env.close()
    
    print("\n训练完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练PPO智能体玩Flappy Bird')
    
    parser.add_argument('--n_episodes', type=int, default=5000, help='训练episode数量')
    parser.add_argument('--update_interval', type=int, default=2048, help='更新策略的步数间隔')
    parser.add_argument('--eval_interval', type=int, default=100, help='评估间隔（episodes）')
    parser.add_argument('--target_score', type=float, default=50, help='目标平均得分')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO裁剪参数')
    parser.add_argument('--value_loss_coef', type=float, default=0.5, help='价值损失系数')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='熵损失系数')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='梯度裁剪阈值')
    parser.add_argument('--n_epochs', type=int, default=10, help='每次更新的训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='训练设备')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志目录')
    parser.add_argument('--model_dir', type=str, default='models', help='模型保存目录')
    
    args = parser.parse_args()
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    train(args)
