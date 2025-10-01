"""
Flappy Bird游戏环境
实现了一个符合OpenAI Gym接口的Flappy Bird环境
"""
import numpy as np
import pygame
from typing import Tuple, Optional


class FlappyBirdEnv:
    """Flappy Bird游戏环境"""
    
    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None):
        self.screen_width = 400
        self.screen_height = 600
        self.bird_x = 100
        self.bird_size = 20
        self.pipe_width = 80
        self.pipe_gap = 200  # 管道之间的垂直间隙
        self.pipe_velocity = 3
        self.gravity = 0.5
        self.jump_velocity = -10
        self.max_velocity = 10
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Flappy Bird")
            self.clock = pygame.time.Clock()
        
        if seed is not None:
            np.random.seed(seed)
            
        self.reset()
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
            
        self.bird_y = self.screen_height // 2
        self.bird_velocity = 0
        
        self.pipes = []
        self._add_pipe()
        
        self.score = 0
        self.steps = 0
        self.done = False
        
        return self._get_state()
    
    def _add_pipe(self):
        """添加新管道"""
        gap_y = np.random.randint(150, self.screen_height - 150 - self.pipe_gap)
        pipe_x = self.screen_width if len(self.pipes) == 0 else self.pipes[-1]['x'] + 250
        
        self.pipes.append({
            'x': pipe_x,
            'gap_y': gap_y,  # 间隙顶部的y坐标
            'scored': False
        })
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行一步
        action: 0 = 不跳, 1 = 跳
        返回: (state, reward, done, info)
        """
        self.steps += 1
        
        if action == 1:
            self.bird_velocity = self.jump_velocity
        
        self.bird_velocity += self.gravity
        self.bird_velocity = min(self.bird_velocity, self.max_velocity)
        self.bird_y += self.bird_velocity
        
        for pipe in self.pipes:
            pipe['x'] -= self.pipe_velocity
        
        self.pipes = [pipe for pipe in self.pipes if pipe['x'] > -self.pipe_width]
        
        if len(self.pipes) == 0 or self.pipes[-1]['x'] < self.screen_width - 250:
            self._add_pipe()
        
        reward = 0.1  # 存活奖励
        
        for pipe in self.pipes:
            if not pipe['scored'] and pipe['x'] + self.pipe_width < self.bird_x:
                pipe['scored'] = True
                self.score += 1
                reward += 10  # 通过管道获得高奖励
        
        done = False
        
        if self.bird_y <= 0 or self.bird_y >= self.screen_height - self.bird_size:
            done = True
            reward = -10  # 碰撞惩罚
        
        for pipe in self.pipes:
            if (self.bird_x + self.bird_size > pipe['x'] and 
                self.bird_x < pipe['x'] + self.pipe_width):
                if (self.bird_y < pipe['gap_y'] or 
                    self.bird_y + self.bird_size > pipe['gap_y'] + self.pipe_gap):
                    done = True
                    reward = -10  # 碰撞惩罚
        
        if len(self.pipes) > 0:
            next_pipe = self.pipes[0]
            pipe_center_y = next_pipe['gap_y'] + self.pipe_gap / 2
            bird_center_y = self.bird_y + self.bird_size / 2
            distance_to_center = abs(bird_center_y - pipe_center_y)
            center_reward = max(0, 0.5 * (1 - distance_to_center / (self.screen_height / 2)))
            reward += center_reward
        
        self.done = done
        
        info = {
            'score': self.score,
            'steps': self.steps
        }
        
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """
        获取当前状态
        状态包含：
        1. 鸟的y坐标（归一化）
        2. 鸟的速度（归一化）
        3. 到下一个管道的水平距离（归一化）
        4. 下一个管道gap顶部的y坐标（归一化）
        5. 下一个管道gap底部的y坐标（归一化）
        6. 鸟与gap中心的垂直距离（归一化）
        """
        if len(self.pipes) == 0:
            return np.array([
                self.bird_y / self.screen_height,
                self.bird_velocity / self.max_velocity,
                1.0,
                0.5,
                0.5,
                0.0
            ], dtype=np.float32)
        
        next_pipe = None
        for pipe in self.pipes:
            if pipe['x'] + self.pipe_width > self.bird_x:
                next_pipe = pipe
                break
        
        if next_pipe is None:
            next_pipe = self.pipes[0]
        
        bird_y_norm = self.bird_y / self.screen_height
        bird_vel_norm = self.bird_velocity / self.max_velocity
        
        pipe_dist_x = (next_pipe['x'] - self.bird_x) / self.screen_width
        pipe_top_y = next_pipe['gap_y'] / self.screen_height
        pipe_bottom_y = (next_pipe['gap_y'] + self.pipe_gap) / self.screen_height
        
        pipe_center_y = next_pipe['gap_y'] + self.pipe_gap / 2
        bird_center_y = self.bird_y + self.bird_size / 2
        vertical_distance = (bird_center_y - pipe_center_y) / self.screen_height
        
        state = np.array([
            bird_y_norm,
            bird_vel_norm,
            pipe_dist_x,
            pipe_top_y,
            pipe_bottom_y,
            vertical_distance
        ], dtype=np.float32)
        
        return state
    
    def render(self):
        """渲染游戏画面"""
        if self.render_mode != "human":
            return
        
        self.screen.fill((135, 206, 235))  # 天空蓝
        
        for pipe in self.pipes:
            pygame.draw.rect(self.screen, (34, 139, 34),  # 绿色
                           (pipe['x'], 0, self.pipe_width, pipe['gap_y']))
            pygame.draw.rect(self.screen, (34, 139, 34),
                           (pipe['x'], pipe['gap_y'] + self.pipe_gap, 
                            self.pipe_width, self.screen_height))
        
        pygame.draw.circle(self.screen, (255, 255, 0),  # 黄色
                          (int(self.bird_x + self.bird_size // 2), 
                           int(self.bird_y + self.bird_size // 2)),
                          self.bird_size // 2)
        
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS
    
    def close(self):
        """关闭环境"""
        if self.screen is not None:
            pygame.quit()
    
    @property
    def observation_space_shape(self):
        """返回观察空间的形状"""
        return (6,)
    
    @property
    def action_space_n(self):
        """返回动作空间的大小"""
        return 2  # 0: 不跳, 1: 跳
