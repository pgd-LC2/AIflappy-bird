# 🤖 Flappy Bird AI - 强化学习自动训练系统

这是一个基于PPO（Proximal Policy Optimization）强化学习算法的Flappy Bird AI训练和演示系统。

## ✨ 特性

- 🚀 **一键运行** - 下载后运行一个命令即可自动训练和演示
- 🧠 **智能训练** - 自动调用全部算力训练15000次迭代
- 🎮 **实时演示** - 训练完成后在浏览器中观看AI玩游戏
- 📊 **性能追踪** - 实时显示得分和训练进度
- 🎯 **高性能** - 训练出的AI可以达到数百分的高分

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/pgd-LC2/AIflappy-bird.git
cd AIflappy-bird
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 一键运行

```bash
python run.py
```

就这么简单！程序会自动：
1. 检查并安装所需依赖
2. 训练AI模型（15000 episodes）
3. 启动Web服务器
4. 在浏览器中打开演示页面

### 4. 观看AI玩游戏

训练完成后，浏览器会自动打开 `http://localhost:8080`

点击 **"开始新游戏"** 按钮，即可观看AI玩Flappy Bird！

## 📋 系统要求

- Python 3.8+
- 推荐使用GPU加速训练（可选）
- 4GB+ RAM
- 现代浏览器（Chrome/Firefox/Safari）

## 🔧 高级选项

### 只训练不启动Web服务器

```bash
python train.py --n_episodes 15000
```

### 只启动Web服务器（使用已训练模型）

```bash
python web_app/app.py
```

然后在浏览器中打开 `http://localhost:8080`

### 评估模型性能

```bash
python evaluate.py --model_path models/best_model.pth --n_eval 30
```

### 自定义训练参数

```bash
python train.py --n_episodes 20000 --lr 5e-4 --update_interval 4096
```

## 📊 训练参数说明

- `--n_episodes`: 训练轮数（默认15000）
- `--lr`: 学习率（默认3e-4）
- `--update_interval`: 策略更新间隔（默认2048）
- `--eval_interval`: 评估间隔（默认100）
- `--seed`: 随机种子（默认42）

## 🎮 Web界面功能

- **开始新游戏** - 开始一局新游戏
- **暂停/继续** - 暂停或继续游戏
- **加速** - 调整游戏速度（1x/2x/4x）
- **实时得分** - 显示当前得分和历史最高分

## 🏆 性能指标

训练完成的AI通常能达到：
- 平均得分：50-100+
- 最高得分：200-300+
- 稳定性：在不同随机种子下表现一致

## 📁 项目结构

```
AIflappy-bird/
├── run.py                  # 一键运行脚本
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── requirements.txt       # 依赖列表
├── src/                   # 源代码
│   ├── game/             # 游戏环境
│   ├── agent/            # PPO智能体
│   └── utils/            # 工具函数
├── web_app/              # Web演示应用
│   ├── app.py           # Flask后端
│   └── templates/       # HTML模板
└── models/               # 训练好的模型
```

## 🧪 技术细节

### 算法

- **PPO (Proximal Policy Optimization)** - 一种稳定高效的策略梯度算法
- **Actor-Critic架构** - 同时学习策略和价值函数
- **GAE (Generalized Advantage Estimation)** - 用于计算优势函数

### 状态空间

6维状态向量：
1. 鸟的y坐标（归一化）
2. 鸟的速度（归一化）
3. 到下一个管道的水平距离
4. 下一个管道gap顶部的y坐标
5. 下一个管道gap底部的y坐标
6. 鸟与gap中心的垂直距离

### 动作空间

2个离散动作：
- 0: 不跳
- 1: 跳

### 奖励函数

- 存活奖励：每步 +0.1
- 通过管道：+10
- 碰撞惩罚：-10
- 中心对齐奖励：根据与管道中心的距离

## 🐛 故障排除

### 端口被占用

如果看到 "Address already in use" 错误：

```bash
# Linux/Mac
lsof -ti:8080 | xargs kill -9

# Windows
netstat -ano | findstr :8080
taskkill /PID <PID> /F
```

### 依赖安装失败

尝试使用国内镜像：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 训练速度慢

- 确保安装了CUDA版本的PyTorch（如果有GPU）
- 减少 `--update_interval` 参数
- 使用更少的 `--n_episodes`

## 📝 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📧 联系方式

如有问题，请在GitHub上提交Issue。

---

⭐ 如果这个项目对你有帮助，欢迎给个Star！
