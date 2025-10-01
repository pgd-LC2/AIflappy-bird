"""
评估训练好的PPO智能体
测试其在不同随机种子下的鲁棒性
"""
import argparse
import os
import numpy as np
import torch
from src.game import FlappyBirdEnv
from src.agent import PPOAgent
from src.utils.visualizer import plot_evaluation_results


def evaluate(args):
    """评估智能体"""
    
    env = FlappyBirdEnv(render_mode=None)
    
    agent = PPOAgent(
        state_dim=env.observation_space_shape[0],
        action_dim=env.action_space_n,
        device=args.device
    )
    
    if not os.path.exists(args.model_path):
        print(f"错误：模型文件不存在: {args.model_path}")
        return
    
    agent.load(args.model_path)
    print(f"✓ 已加载模型: {args.model_path}\n")
    
    print("="*60)
    print("开始评估")
    print("="*60)
    print(f"评估轮数: {args.n_eval}")
    print(f"使用设备: {args.device}")
    print(f"确定性策略: {args.deterministic}")
    print("="*60 + "\n")
    
    all_scores = []
    all_rewards = []
    all_lengths = []
    seeds_used = []
    
    for i in range(args.n_eval):
        if args.random_seed:
            seed = np.random.randint(0, 10000)
        else:
            seed = args.seed + i if args.seed is not None else None
        
        seeds_used.append(seed)
        
        state = env.reset(seed=seed)
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _, _ = agent.select_action(state, deterministic=args.deterministic)
            
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        score = info['score']
        all_scores.append(score)
        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)
        
        print(f"测试 {i+1}/{args.n_eval}: 得分={score}, 奖励={episode_reward:.2f}, "
              f"长度={episode_length}, 种子={seed}")
    
    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    max_score = np.max(all_scores)
    min_score = np.min(all_scores)
    median_score = np.median(all_scores)
    
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    print(f"平均得分: {mean_score:.2f} ± {std_score:.2f}")
    print(f"中位数得分: {median_score:.2f}")
    print(f"最高得分: {max_score}")
    print(f"最低得分: {min_score}")
    print(f"得分范围: [{min_score}, {max_score}]")
    print(f"\n平均奖励: {np.mean(all_rewards):.2f}")
    print(f"平均长度: {np.mean(all_lengths):.2f}")
    
    success_count = sum(1 for s in all_scores if s >= 50)
    success_rate = success_count / len(all_scores) * 100
    print(f"\n达标率 (得分≥50): {success_rate:.1f}% ({success_count}/{len(all_scores)})")
    
    if std_score < 10:
        robustness = "优秀"
    elif std_score < 20:
        robustness = "良好"
    elif std_score < 30:
        robustness = "一般"
    else:
        robustness = "较差"
    
    print(f"鲁棒性评估: {robustness} (标准差: {std_score:.2f})")
    
    print("="*60 + "\n")
    
    if args.save_results:
        results_file = os.path.join('logs', 'evaluation_results.txt')
        with open(results_file, 'w') as f:
            f.write("评估结果\n")
            f.write("="*60 + "\n")
            f.write(f"模型: {args.model_path}\n")
            f.write(f"评估轮数: {args.n_eval}\n")
            f.write(f"确定性策略: {args.deterministic}\n\n")
            
            f.write(f"平均得分: {mean_score:.2f} ± {std_score:.2f}\n")
            f.write(f"中位数得分: {median_score:.2f}\n")
            f.write(f"最高得分: {max_score}\n")
            f.write(f"最低得分: {min_score}\n")
            f.write(f"达标率 (≥50分): {success_rate:.1f}%\n")
            f.write(f"鲁棒性: {robustness}\n\n")
            
            f.write("详细结果:\n")
            for i, (score, seed) in enumerate(zip(all_scores, seeds_used)):
                f.write(f"  测试 {i+1}: 得分={score}, 种子={seed}\n")
        
        print(f"✓ 评估结果已保存到: {results_file}")
    
    if args.plot:
        plot_path = os.path.join('logs', 'evaluation_plot.png')
        plot_evaluation_results(all_scores, seeds_used, save_path=plot_path)
    
    env.close()
    
    return mean_score >= 50


def evaluate_with_render(args):
    """以可视化方式评估智能体"""
    
    env = FlappyBirdEnv(render_mode='human')
    
    agent = PPOAgent(
        state_dim=env.observation_space_shape[0],
        action_dim=env.action_space_n,
        device=args.device
    )
    
    if not os.path.exists(args.model_path):
        print(f"错误：模型文件不存在: {args.model_path}")
        return
    
    agent.load(args.model_path)
    print(f"✓ 已加载模型: {args.model_path}\n")
    print("按任意键开始游戏...\n")
    
    for episode in range(args.n_eval):
        print(f"\n第 {episode + 1}/{args.n_eval} 局")
        
        seed = np.random.randint(0, 10000) if args.random_seed else args.seed
        state = env.reset(seed=seed)
        episode_reward = 0
        done = False
        
        while not done:
            env.render()
            
            action, _, _ = agent.select_action(state, deterministic=args.deterministic)
            
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_reward += reward
        
        print(f"本局得分: {info['score']}, 累计奖励: {episode_reward:.2f}")
        
        if episode < args.n_eval - 1:
            import time
            time.sleep(2)  # 暂停2秒再开始下一局
    
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估PPO智能体')
    
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                       help='模型文件路径')
    parser.add_argument('--n_eval', type=int, default=20,
                       help='评估轮数')
    parser.add_argument('--deterministic', action='store_true',
                       help='使用确定性策略（选择概率最大的动作）')
    parser.add_argument('--seed', type=int, default=42,
                       help='基础随机种子')
    parser.add_argument('--random_seed', action='store_true',
                       help='使用随机种子（测试鲁棒性）')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='运行设备')
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='保存评估结果')
    parser.add_argument('--plot', action='store_true', default=True,
                       help='绘制评估结果图表')
    parser.add_argument('--render', action='store_true',
                       help='可视化评估（显示游戏画面）')
    
    args = parser.parse_args()
    
    os.makedirs('logs', exist_ok=True)
    
    if args.render:
        evaluate_with_render(args)
    else:
        evaluate(args)
