#!/usr/bin/env python3
"""
Flappy Bird AI - 一键训练和演示程序
运行此文件即可自动训练AI并在浏览器中展示
"""
import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """检查并安装依赖"""
    print("=" * 60)
    print("🔍 检查依赖...")
    print("=" * 60)
    
    try:
        import torch
        import pygame
        import numpy as np
        import flask
        import flask_cors
        print("✅ 所有依赖已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("\n正在安装依赖...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 依赖安装完成")
        return True

def train_model():
    """训练AI模型"""
    print("\n" + "=" * 60)
    print("🚀 开始训练AI (15000 episodes)")
    print("=" * 60)
    print("训练将调用全部算力，预计需要一些时间...")
    print("请耐心等待，训练过程中会显示实时进度\n")
    
    model_path = Path("models/best_model.pth")
    if model_path.exists():
        response = input("\n发现已有训练模型，是否重新训练? (y/N): ")
        if response.lower() != 'y':
            print("✅ 使用现有模型")
            return True
    
    cmd = [
        sys.executable, "train.py",
        "--n_episodes", "15000",
        "--update_interval", "2048",
        "--eval_interval", "100",
        "--lr", "3e-4",
        "--seed", "42"
    ]
    
    try:
        subprocess.check_call(cmd)
        print("\n✅ 训练完成！")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ 训练过程中出现错误")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        return False

def start_web_server():
    """启动Web服务器"""
    print("\n" + "=" * 60)
    print("🌐 启动Web服务器...")
    print("=" * 60)
    
    model_path = Path("models/best_model.pth")
    if not model_path.exists():
        model_path = Path("super_models/best_model.pth")
        if not model_path.exists():
            print("❌ 错误: 未找到训练好的模型文件")
            print("请先运行训练或确保模型文件存在")
            return False
    
    import subprocess
    print("正在启动Flask服务器...")
    print("如果看到错误信息，请仔细阅读以了解问题所在\n")
    
    web_process = subprocess.Popen(
        [sys.executable, "web_app/app.py"]
    )
    
    print("\n等待服务器启动...")
    time.sleep(5)
    
    if web_process.poll() is not None:
        print("\n❌ 服务器启动失败！")
        print("请查看上面的错误信息")
        return False
    
    url = "http://localhost:8080"
    print(f"\n✅ 服务器已启动: {url}")
    print("\n" + "=" * 60)
    print("🎮 AI Flappy Bird 已准备就绪！")
    print("=" * 60)
    print(f"\n浏览器将自动打开 {url}")
    print("点击'开始新游戏'按钮即可观看AI玩游戏")
    print("\n按 Ctrl+C 退出程序\n")
    
    webbrowser.open(url)
    
    try:
        web_process.wait()
    except KeyboardInterrupt:
        print("\n\n👋 感谢使用！程序已退出")
        web_process.terminate()
        web_process.wait(timeout=5)
    
    return True

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("🤖 Flappy Bird AI - 自动训练和演示系统")
    print("=" * 60)
    
    os.chdir(Path(__file__).parent)
    
    if not check_dependencies():
        print("❌ 依赖检查失败")
        sys.exit(1)
    
    if not train_model():
        response = input("\n是否仍要启动Web服务器? (y/N): ")
        if response.lower() != 'y':
            print("程序退出")
            sys.exit(1)
    
    start_web_server()

if __name__ == "__main__":
    main()
