"""
超强PPO智能体训练脚本 - 目标: 100,000+ 分
"""
import argparse
import os
import numpy as np
import torch
from src.game import FlappyBirdEnv
from src.agent import PPOAgent
from src.utils import Logger


def train_super_ai(args):
    """训练超强PPO智能体"""
    
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    env = FlappyBirdEnv(render_mode=None, seed=args.seed)
    
    agent = PPOAgent(
        state_dim=env.observation_space_shape[0],
        action_dim=env.action_space_n,
        lr=args.lr,
        gamma=args.gamma,  # 提高gamma，更重视长期奖励
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,  # 降低熵系数，减少随机性
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=args.device
    )
    
    logger = Logger(log_dir=args.log_dir)
    
    print("=" * 80)
    print("🚀 开始训练超强PPO智能体 - 目标: 100,000+ 分")
    print("=" * 80)
    print(f"训练Episodes: {args.n_episodes}")
    print(f"更新间隔: {args.update_interval} 步")
    print(f"学习率: {args.lr}")
    print(f"Gamma (折扣因子): {args.gamma}")
    print(f"网络规模: 增强版 (512 hidden units)")
    print(f"设备: {args.device}")
    print(f"随机种子: {args.seed if args.seed is not None else '未设置'}")
    print("=" * 80 + "\n")
    
    best_avg_score = 0
    episode = 0
    consecutive_high_scores = 0  # 连续高分计数
    
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
        
        score = info['score']
        logger.log_episode(episode, episode_reward, episode_length, score)
        
        if score >= 1000:
            consecutive_high_scores += 1
            print(f"🎯 高分警报! Episode {episode}: {score}分 (连续{consecutive_high_scores}次)")
        else:
            consecutive_high_scores = 0
        
        if episode % args.eval_interval == 0:
            recent_perf = logger.get_recent_performance(100)
            avg_score = recent_perf['avg_score']
            
            print(f"\n{'=' * 80}")
            print(f"🔍 Episode {episode} 详细评估")
            print(f"{'=' * 80}")
            print(f"最近100个Episode平均得分: {avg_score:.2f}")
            print(f"最近100个Episode最高得分: {recent_perf['max_score']}")
            print(f"最近100个Episode平均奖励: {recent_perf['avg_reward']:.2f}")
            print(f"当前Episode得分: {score}")
            print(f"连续高分(≥1000)次数: {consecutive_high_scores}")
            
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                model_name = "best_model"
                
                if avg_score >= 10000:
                    model_name = "legendary_model"
                elif avg_score >= 5000:
                    model_name = "master_model"
                elif avg_score >= 1000:
                    model_name = "expert_model"
                elif avg_score >= 100:
                    model_name = "advanced_model"
                
                model_path = os.path.join(args.model_dir, f'{model_name}.pth')
                agent.save(model_path)
                print(f"✨ 保存{model_name} (平均得分: {avg_score:.2f}) 到 {model_path}")
            
            print(f"{'=' * 80}\n")
            
            if episode % (args.eval_interval * 5) == 0:
                checkpoint_path = os.path.join(args.model_dir, f'super_checkpoint_ep{episode}.pth')
                agent.save(checkpoint_path)
                print(f"💾 保存检查点到 {checkpoint_path}\n")
        
        if episode >= 500:  # 至少训练500个episode
            recent_perf = logger.get_recent_performance(200)
            avg_score = recent_perf['avg_score']
            
            if avg_score >= 100000:
                print(f"\n🏆 传奇级成就达成！")
                print(f"平均得分: {avg_score:.2f} >= 100,000")
                break
            elif avg_score >= 10000:
                print(f"\n🥇 大师级成就达成！")
                print(f"平均得分: {avg_score:.2f} >= 10,000")
                if episode >= 2000:  # 大师级后再训练一段时间
                    break
            elif avg_score >= 1000:
                print(f"\n🥈 专家级成就达成！")
                print(f"平均得分: {avg_score:.2f} >= 1,000")
            elif avg_score >= 100:
                print(f"\n🥉 高级成就达成！")
                print(f"平均得分: {avg_score:.2f} >= 100")
    
    final_model_path = os.path.join(args.model_dir, 'super_final_model.pth')
    agent.save(final_model_path)
    print(f"\n💎 保存最终超强模型到 {final_model_path}")
    
    logger.save()
    logger.print_summary()
    
    env.close()
    
    print(f"\n🎊 超强AI训练完成！共训练 {episode} episodes")
    return recent_perf['avg_score'] if episode >= 500 else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练超强PPO智能体 (目标: 100,000+ 分)')
    
    parser.add_argument('--n_episodes', type=int, default=20000, help='训练episode数量')
    parser.add_argument('--update_interval', type=int, default=4096, help='更新策略的步数间隔（增大以获得更多经验）')
    parser.add_argument('--eval_interval', type=int, default=100, help='评估间隔（episodes）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率（降低以提高稳定性）')
    parser.add_argument('--gamma', type=float, default=0.999, help='折扣因子（提高以重视长期奖励）')
    parser.add_argument('--gae_lambda', type=float, default=0.98, help='GAE lambda（提高以减少偏差）')
    parser.add_argument('--clip_epsilon', type=float, default=0.1, help='PPO裁剪参数（降低以提高稳定性）')
    parser.add_argument('--value_loss_coef', type=float, default=0.5, help='价值损失系数')
    parser.add_argument('--entropy_coef', type=float, default=0.005, help='熵损失系数（降低以减少随机性）')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='梯度裁剪阈值')
    parser.add_argument('--n_epochs', type=int, default=15, help='每次更新的训练轮数（增加以充分利用数据）')
    parser.add_argument('--batch_size', type=int, default=128, help='批量大小（增大以提高稳定性）')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='训练设备')
    parser.add_argument('--log_dir', type=str, default='super_logs', help='日志目录')
    parser.add_argument('--model_dir', type=str, default='super_models', help='模型保存目录')
    
    args = parser.parse_args()
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    final_score = train_super_ai(args)
    
    if final_score >= 100000:
        print("\n🎆 恭喜! AI已达到传奇级水平 (100,000+ 分)!")
    elif final_score >= 10000:
        print("\n🎇 恭喜! AI已达到大师级水平 (10,000+ 分)!")
    elif final_score >= 1000:
        print("\n🎈 恭喜! AI已达到专家级水平 (1,000+ 分)!")
    else:
        print(f"\n📈 AI训练完成，最终平均分: {final_score:.2f}")
