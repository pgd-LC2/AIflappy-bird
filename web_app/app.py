"""
Flask Web应用 - AI玩Flappy Bird展示
"""
from flask import Flask, render_template, jsonify
from flask_cors import CORS
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game import FlappyBirdEnv
from src.agent import PPOAgent

app = Flask(__name__)
CORS(app)

game_state = {
    'env': None,
    'agent': None,
    'current_state': None,
    'score': 0,
    'is_playing': False,
    'game_over': False
}

def initialize_game():
    """初始化游戏和AI"""
    if game_state['env'] is None:
        game_state['env'] = FlappyBirdEnv(render_mode=None)
        
        agent = PPOAgent(
            state_dim=game_state['env'].observation_space_shape[0],
            action_dim=game_state['env'].action_space_n,
            device='cpu'
        )
        
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_model.pth')
        agent.load(model_path)
        game_state['agent'] = agent
        
    game_state['current_state'] = game_state['env'].reset()
    game_state['score'] = 0
    game_state['is_playing'] = True
    game_state['game_over'] = False

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_game():
    """开始新游戏"""
    initialize_game()
    return jsonify({
        'success': True,
        'message': 'Game started'
    })

@app.route('/api/step', methods=['GET'])
def game_step():
    """执行一步游戏"""
    if not game_state['is_playing'] or game_state['game_over']:
        return jsonify({
            'game_over': True,
            'score': game_state['score']
        })
    
    action, _, _ = game_state['agent'].select_action(
        game_state['current_state'], 
        deterministic=True
    )
    
    next_state, reward, done, info = game_state['env'].step(action)
    game_state['current_state'] = next_state
    game_state['score'] = info['score']
    
    if done:
        game_state['game_over'] = True
        game_state['is_playing'] = False
    
    game_data = {
        'bird_y': float(game_state['env'].bird_y),
        'bird_velocity': float(game_state['env'].bird_velocity),
        'pipes': [],
        'score': game_state['score'],
        'game_over': game_state['game_over'],
        'action': int(action)
    }
    
    for pipe in game_state['env'].pipes:
        game_data['pipes'].append({
            'x': float(pipe['x']),
            'gap_y': float(pipe['gap_y']),
            'gap_size': float(game_state['env'].pipe_gap)
        })
    
    return jsonify(game_data)

@app.route('/api/state', methods=['GET'])
def get_state():
    """获取当前游戏状态"""
    if game_state['env'] is None:
        return jsonify({
            'initialized': False
        })
    
    return jsonify({
        'initialized': True,
        'is_playing': game_state['is_playing'],
        'game_over': game_state['game_over'],
        'score': game_state['score']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
