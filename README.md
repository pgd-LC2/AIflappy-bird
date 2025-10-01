# AIflappy-bird

使用PPO（Proximal Policy Optimization）强化学习算法训练AI智能体玩Flappy Bird游戏。

## 🎯 项目目标

训练一个稳定的PPO智能体，使其在Flappy Bird游戏中能够：
- **平均得分 > 50分**（通过50对管道）
- **具备良好的鲁棒性**：在不同随机种子下表现稳定
- **学会合理的游戏策略**：不依赖特定随机种子

## 🏗️ 项目结构

```
AIflappy-bird/
├── src/
│   ├── game/
│   │   ├── __init__.py
│   │   └── flappy_bird_env.py      # Flappy Bird游戏环境
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── ppo_network.py          # Actor-Critic神经网络
│   │   └── ppo_agent.py            # PPO智能体实现
│   └── utils/
│       ├── __init__.py
│       ├── logger.py               # 训练日志记录
│       └── visualizer.py           # 数据可视化工具
├── train.py                        # 训练脚本
├── evaluate.py                     # 评估脚本
├── visualize.py                    # 可视化脚本
├── requirements.txt                # 依赖包列表
├── models/                         # 保存训练好的模型
├── logs/                          # 训练日志和图表
└── README.md                      # 项目文档
```

## 🚀 快速开始

### 1. 环境设置

```bash
# 克隆项目
git clone https://github.com/pgd-LC2/AIflappy-bird.git
cd AIflappy-bird

# 安装依赖
pip install -r requirements.txt
```

### 2. 开始训练

```bash
# 基础训练（使用默认参数）
python train.py

# 自定义训练参数
python train.py --n_episodes 3000 --lr 5e-4 --seed 42

# 查看所有可用参数
python train.py --help
```

### 3. 评估模型

```bash
# 评估最佳模型（20轮测试，不同随机种子）
python evaluate.py --model_path models/best_model.pth --n_eval 20 --random_seed

# 可视化评估（观看AI玩游戏）
python evaluate.py --model_path models/best_model.pth --render --n_eval 5
```

### 4. 可视化训练结果

```bash
# 生成训练曲线图
python visualize.py
```

## 🧠 算法特点

### PPO (Proximal Policy Optimization)
- **策略梯度方法**：直接优化策略网络
- **Actor-Critic架构**：结合策略网络和价值网络
- **裁剪目标函数**：防止策略更新过大，提高训练稳定性
- **GAE (广义优势估计)**：减少方差，提高学习效率

### 网络架构
```
输入层 (6维状态) → 共享特征层 (128->128) → 分支：
                                           ├── Actor网络 → 动作概率
                                           └── Critic网络 → 状态价值
```

### 状态特征 (6维)
1. 鸟的y坐标（归一化）
2. 鸟的速度（归一化）
3. 到下一个管道的水平距离（归一化）
4. 下一个管道gap顶部的y坐标（归一化）
5. 下一个管道gap底部的y坐标（归一化）
6. 鸟与gap中心的垂直距离（归一化）

### 奖励机制
- **存活奖励**: +0.1（每步）
- **通过管道**: +10
- **接近管道中心**: +0.5（距离越近奖励越高）
- **碰撞惩罚**: -10

## 📊 训练监控

训练过程中会实时显示：
- Episode得分、奖励、长度
- 最近100个Episode的平均性能
- 策略损失、价值损失、熵损失
- 模型保存信息

### 可视化输出
- **训练曲线**: 得分、奖励、Episode长度随时间变化
- **评估结果**: 不同随机种子下的性能分布
- **统计信息**: 平均值、标准差、达标率

## ⚙️ 主要参数

### 训练参数
- `--n_episodes`: 训练episode数量 (默认: 5000)
- `--update_interval`: 更新策略的步数间隔 (默认: 2048)
- `--target_score`: 目标平均得分 (默认: 50)
- `--seed`: 随机种子

### PPO超参数
- `--lr`: 学习率 (默认: 3e-4)
- `--gamma`: 折扣因子 (默认: 0.99)
- `--gae_lambda`: GAE lambda (默认: 0.95)
- `--clip_epsilon`: PPO裁剪参数 (默认: 0.2)
- `--entropy_coef`: 熵损失系数 (默认: 0.01)

## 🎮 使用示例

### 完整训练流程
```bash
# 1. 训练模型
python train.py --n_episodes 3000 --seed 42

# 2. 评估性能
python evaluate.py --model_path models/best_model.pth --n_eval 50 --random_seed

# 3. 可视化结果
python visualize.py

# 4. 观看AI玩游戏
python evaluate.py --model_path models/best_model.pth --render --n_eval 3
```

### 多种子训练（提高鲁棒性）
```bash
# 使用不同随机种子训练多个模型
for seed in 42 123 456 789; do
    python train.py --seed $seed --n_episodes 2000
done
```

## 🔧 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   python train.py --device cpu
   ```

2. **训练过慢**
   ```bash
   python train.py --update_interval 1024 --batch_size 32
   ```

3. **收敛困难**
   ```bash
   python train.py --lr 1e-4 --entropy_coef 0.02
   ```

### 性能优化建议

- **调整学习率**: 从3e-4开始，如果收敛慢可降低到1e-4
- **修改奖励函数**: 在`flappy_bird_env.py`中调整奖励权重
- **增加训练步数**: 延长`update_interval`以收集更多经验
- **调节网络大小**: 在`ppo_network.py`中修改`hidden_dim`

## 📈 预期结果

一个训练良好的模型应该能够：
- 在大多数随机种子下得分 > 50
- 标准差 < 20（表示性能稳定）
- 达标率 > 80%（50分以上的比例）

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件
