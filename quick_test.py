"""快速测试脚本"""
import torch
import numpy as np
from src.game import FlappyBirdEnv
from src.agent import PPOAgent

env = FlappyBirdEnv(render_mode=None)
agent = PPOAgent(
    state_dim=env.observation_space_shape[0],
    action_dim=env.action_space_n,
    device='cpu'
)

agent.load('models/best_model.pth')
print("✓ 模型加载成功\n")

scores = []
for i in range(10):
    state = env.reset(seed=np.random.randint(10000))
    done = False
    while not done:
        action, _, _ = agent.select_action(state, deterministic=True)
        state, _, done, info = env.step(action)
    scores.append(info['score'])
    print(f"测试 {i+1}: 得分={info['score']}")

print(f"\n平均得分: {np.mean(scores):.2f}")
print(f"最高得分: {max(scores)}")
print(f"最低得分: {min(scores)}")
