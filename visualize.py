"""
可视化训练结果
"""
import argparse
import os
import glob
from src.utils.visualizer import plot_training_curves


def visualize_training(args):
    """可视化训练曲线"""
    
    if args.log_file:
        if not os.path.exists(args.log_file):
            print(f"错误：日志文件不存在: {args.log_file}")
            return
        log_files = [args.log_file]
    else:
        log_files = glob.glob(os.path.join(args.log_dir, 'training_*.json'))
        
        if not log_files:
            print(f"错误：在 {args.log_dir} 目录下没有找到日志文件")
            return
        
        log_files.sort(key=os.path.getmtime, reverse=True)
        print(f"找到 {len(log_files)} 个日志文件")
        print(f"使用最新的日志文件: {log_files[0]}\n")
    
    for log_file in log_files[:1]:  # 只处理第一个（最新的）
        save_path = os.path.join(args.output_dir, 'training_curves.png')
        plot_training_curves(log_file, save_path)
        print(f"\n✓ 训练曲线已生成")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可视化训练结果')
    
    parser.add_argument('--log_file', type=str, default=None,
                       help='指定日志文件路径（可选，默认使用最新的）')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='日志目录')
    parser.add_argument('--output_dir', type=str, default='logs',
                       help='输出目录')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    visualize_training(args)
