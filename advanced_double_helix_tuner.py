# -*- coding: utf-8 -*-
# advanced_double_helix_tuner.py
# 高級調控面板：三種進化模式

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from collections import deque
from enum import Enum

class EvolutionMode(Enum):
    """三種進化策略"""
    SYMMETRY_SEEKING = "對稱追求模式"      # 鼓勵形成對稱解
    LONG_EXPLORATION = "長期探索模式"      # 持續破對稱探索
    PHASE_BIFURCATION = "相位分岔模式"     # 研究相位動力學

# ===== 可調控的雙螺旋網絡 =====
class TunableCounterRotatingNetwork(nn.Module):
    def __init__(self, feature_dim=32, hidden_dim=64, action_dim=64, 
                 learnable_phase_velocity=False):
        super().__init__()
        
        self.cw_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.cw_actor = nn.Linear(hidden_dim, action_dim)
        
        self.ccw_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.ccw_actor = nn.Linear(hidden_dim, action_dim)
        
        self.value_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 相位參數
        self.cw_phase = nn.Parameter(torch.tensor(0.0))
        self.ccw_phase = nn.Parameter(torch.tensor(math.pi))
        
        # 🔧 可學習的相位速度（用於分岔研究）
        if learnable_phase_velocity:
            self.phase_velocity = nn.Parameter(torch.tensor(0.1))
            self.phase_noise_scale = nn.Parameter(torch.tensor(0.01))
        else:
            self.register_buffer('phase_velocity', torch.tensor(0.1))
            self.register_buffer('phase_noise_scale', torch.tensor(0.0))
    
    def forward(self, state, temperature=1.0, step=0, add_phase_noise=False):
        t = (step % 200) / 200.0
        
        # 動態相位演化
        cw_phase_current = self.cw_phase + \
                          self.phase_velocity * math.sin(2 * math.pi * t) + \
                          0.3 * math.cos(4 * math.pi * t)
        ccw_phase_current = self.ccw_phase - \
                           self.phase_velocity * math.sin(2 * math.pi * t) - \
                           0.3 * math.cos(4 * math.pi * t)
        
        # 🔧 相位噪聲（研究分岔時啟用）
        if add_phase_noise:
            cw_phase_current = cw_phase_current + \
                torch.randn_like(cw_phase_current) * self.phase_noise_scale
            ccw_phase_current = ccw_phase_current + \
                torch.randn_like(ccw_phase_current) * self.phase_noise_scale
        
        cw_features = self.cw_encoder(state)
        ccw_features = self.ccw_encoder(state)
        
        cw_logits = self.cw_actor(cw_features)
        ccw_logits = self.ccw_actor(ccw_features)
        
        cw_rotation = torch.cos(cw_phase_current) * 0.5 + 0.5
        ccw_rotation = torch.sin(ccw_phase_current) * 0.5 + 0.5
        
        entropy_bonus = temperature * 0.1
        
        cw_probs = torch.softmax((cw_logits + entropy_bonus) / temperature, dim=-1)
        ccw_probs = torch.softmax((ccw_logits + entropy_bonus) / temperature, dim=-1)
        
        coupling_strength = torch.sigmoid(torch.tensor(1.0 - temperature))
        
        mixed_cw = cw_rotation * cw_probs + (1 - cw_rotation) * ccw_probs
        mixed_ccw = ccw_rotation * ccw_probs + (1 - ccw_rotation) * cw_probs
        
        mixed_probs = torch.stack([mixed_cw, mixed_ccw])
        
        fused = torch.cat([cw_features, ccw_features], dim=-1)
        state_value = self.value_fusion(fused)
        
        return mixed_probs, state_value, {
            'cw_phase': cw_phase_current.item(),
            'ccw_phase': ccw_phase_current.item(),
            'coupling': coupling_strength.item(),
            'phase_velocity': self.phase_velocity.item()
        }

# ===== 可配置環境 =====
class ConfigurableHelixEnv:
    def __init__(self, max_steps=100, mode=EvolutionMode.LONG_EXPLORATION):
        self.max_steps = max_steps
        self.mode = mode
        self.reset()
    
    def reset(self):
        self.cw_state = np.random.randint(1, 65)
        self.ccw_state = np.random.randint(1, 65)
        self.cw_history = [self.cw_state]
        self.ccw_history = [self.ccw_state]
        self.step_count = 0
        self.stagnation_counter = 0
        return self._get_state()
    
    def _get_state(self):
        features = np.zeros(32)
        features[0] = (self.cw_state - 1) / 63.0
        features[1] = (self.ccw_state - 1) / 63.0
        features[2] = 1.0 - abs(self.cw_state + self.ccw_state - 65) / 64.0
        
        xor_val = (self.cw_state - 1) ^ (self.ccw_state - 1)
        features[3] = bin(xor_val).count('1') / 6.0
        
        if len(self.cw_history) > 1:
            features[4] = (self.cw_history[-2] - 1) / 63.0
            features[5] = (self.ccw_history[-2] - 1) / 63.0
        
        unique_cw = len(set(self.cw_history))
        unique_ccw = len(set(self.ccw_history))
        features[6] = unique_cw / len(self.cw_history)
        features[7] = unique_ccw / len(self.ccw_history)
        features[8] = min(self.stagnation_counter / 10.0, 1.0)
        
        cw_angle = 2 * math.pi * self.cw_state / 64
        ccw_angle = 2 * math.pi * self.ccw_state / 64
        features[9] = math.sin(cw_angle)
        features[10] = math.cos(cw_angle)
        features[11] = math.sin(ccw_angle)
        features[12] = math.cos(ccw_angle)
        features[13] = math.sin(cw_angle - ccw_angle)
        features[14] = math.cos(cw_angle - ccw_angle)
        features[15] = self.step_count / self.max_steps
        
        return torch.FloatTensor(features)
    
    def step(self, cw_action, ccw_action):
        old_cw = self.cw_state
        old_ccw = self.ccw_state
        
        self.cw_state = cw_action + 1
        self.ccw_state = ccw_action + 1
        self.cw_history.append(self.cw_state)
        self.ccw_history.append(self.ccw_state)
        self.step_count += 1
        
        if self.cw_state == old_cw and self.ccw_state == old_ccw:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
        
        reward = self._compute_reward(old_cw, old_ccw)
        done = self.step_count >= self.max_steps
        
        return self._get_state(), reward, done
    
    def _compute_reward(self, old_cw, old_ccw):
        """根據模式配置不同的獎勵函數"""
        reward = 0.0
        
        # === 基礎獎勵（所有模式共享）===
        is_new_pair = (self.cw_state, self.ccw_state) not in \
                      zip(self.cw_history[:-1], self.ccw_history[:-1])
        if is_new_pair:
            reward += 3.0
        
        cw_moved = abs(self.cw_state - old_cw) > 0
        ccw_moved = abs(self.ccw_state - old_ccw) > 0
        if cw_moved or ccw_moved:
            reward += 1.0
        
        # 停滯懲罰
        if self.stagnation_counter >= 5:
            reward -= 5.0 * (self.stagnation_counter - 4)
        
        # === 模式特定獎勵 ===
        if self.mode == EvolutionMode.SYMMETRY_SEEKING:
            # 🎯 強化對稱性獎勵
            symmetry_error = abs(self.cw_state + self.ccw_state - 65)
            if symmetry_error == 0:
                reward += 5.0  # 大幅提高
            elif symmetry_error <= 5:
                reward += 2.0 / (symmetry_error + 1)
            
            # 對稱破缺懲罰（與長期探索相反）
            if symmetry_error > 15:
                reward -= 1.0
        
        elif self.mode == EvolutionMode.LONG_EXPLORATION:
            # 🚀 持續探索獎勵
            cw_delta = self.cw_state - old_cw
            ccw_delta = self.ccw_state - old_ccw
            
            if cw_delta * ccw_delta < 0:
                reward += 2.0
            
            jump_distance = abs(cw_delta) + abs(ccw_delta)
            if jump_distance > 10:
                reward += 1.5 * (jump_distance / 64.0)
            
            # 晚期對稱懲罰（維持探索）
            if self.step_count > 50:
                symmetry_error = abs(self.cw_state + self.ccw_state - 65)
                if symmetry_error == 0:
                    reward -= 2.0  # 懲罰晚期對稱
        
        elif self.mode == EvolutionMode.PHASE_BIFURCATION:
            # 🌀 相位關係獎勵
            cw_angle = 2 * math.pi * self.cw_state / 64
            ccw_angle = 2 * math.pi * self.ccw_state / 64
            phase_diff = abs(cw_angle - ccw_angle)
            
            # 獎勵特定相位關係（0, π/2, π, 3π/2）
            for target in [0, math.pi/2, math.pi, 3*math.pi/2]:
                if abs(phase_diff - target) < 0.3:
                    reward += 3.0
                    break
            
            # 獎勵相位跳躍
            if len(self.cw_history) >= 2:
                old_phase_diff = abs(
                    2 * math.pi * old_cw / 64 - 
                    2 * math.pi * old_ccw / 64
                )
                phase_change = abs(phase_diff - old_phase_diff)
                if phase_change > math.pi / 4:
                    reward += 2.0
        
        # 多樣性獎勵（所有模式）
        if len(self.cw_history) >= 10:
            recent_unique = len(set(self.cw_history[-10:]))
            reward += (recent_unique / 10.0) * 2.0
        
        return reward

# ===== 高級訓練器 =====
class AdvancedA2CTrainer:
    def __init__(self, model, lr=3e-4, gamma=0.99, mode=EvolutionMode.LONG_EXPLORATION):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.mode = mode
        self.global_step = 0
        
        # 🔧 根據模式配置溫度策略
        if mode == EvolutionMode.SYMMETRY_SEEKING:
            self.temperature = 1.5
            self.min_temperature = 0.3
            self.max_temperature = 2.0
        elif mode == EvolutionMode.LONG_EXPLORATION:
            self.temperature = 2.0
            self.min_temperature = 0.8  # 提高下限！
            self.max_temperature = 3.0
        else:  # PHASE_BIFURCATION
            self.temperature = 2.5
            self.min_temperature = 1.0
            self.max_temperature = 4.0
        
        self.temperature_momentum = 0.0
    
    def adjust_temperature(self, diversity_score, phase_diff):
        if self.mode == EvolutionMode.SYMMETRY_SEEKING:
            # 對稱模式：低多樣性時不急著升溫
            if diversity_score < 0.4:
                self.temperature_momentum += 0.03
            elif diversity_score > 0.8:
                self.temperature_momentum -= 0.05
        
        elif self.mode == EvolutionMode.LONG_EXPLORATION:
            # 🔥 探索模式：保持高溫，防止貼底
            if diversity_score < 0.5 or phase_diff < 2.0:
                self.temperature_momentum += 0.08  # 更強的升溫
            elif diversity_score > 0.95 and phase_diff > 2.8:
                self.temperature_momentum -= 0.02  # 較弱的降溫
            
            # 🔧 添加熵地板（entropy floor）
            if self.temperature < self.min_temperature + 0.3:
                self.temperature_momentum += 0.05
        
        else:  # PHASE_BIFURCATION
            # 分岔模式：根據相位變化調整
            if phase_diff < 1.5 or phase_diff > 4.5:
                self.temperature_momentum += 0.1
            else:
                self.temperature_momentum -= 0.03
        
        self.temperature += self.temperature_momentum
        self.temperature = np.clip(
            self.temperature, 
            self.min_temperature, 
            self.max_temperature
        )
        self.temperature_momentum *= 0.9
    
    def train_episode(self, env):
        state = env.reset()
        states, actions, rewards, values = [], [], [], []
        episode_reward = 0
        meta_info = []
        
        done = False
        while not done:
            with torch.no_grad():
                add_noise = (self.mode == EvolutionMode.PHASE_BIFURCATION)
                probs, state_value, info = self.model(
                    state.unsqueeze(0),
                    temperature=self.temperature,
                    step=self.global_step,
                    add_phase_noise=add_noise
                )
            
            meta_info.append(info)
            
            cw_dist = torch.distributions.Categorical(probs[0])
            ccw_dist = torch.distributions.Categorical(probs[1])
            cw_action = cw_dist.sample().item()
            ccw_action = ccw_dist.sample().item()
            
            next_state, reward, done = env.step(cw_action, ccw_action)
            
            states.append(state)
            actions.append([cw_action, ccw_action])
            rewards.append(reward)
            values.append(state_value)
            
            episode_reward += reward
            state = next_state
            self.global_step += 1
        
        unique_pairs = len(set(zip(env.cw_history, env.ccw_history)))
        diversity_score = unique_pairs / len(env.cw_history)
        avg_phase_diff = np.mean([abs(m['cw_phase'] - m['ccw_phase']) 
                                  for m in meta_info])
        self.adjust_temperature(diversity_score, avg_phase_diff)
        
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns).unsqueeze(1)
        values = torch.cat(values)
        advantages = returns - values.detach()
        
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0
        
        for i, state in enumerate(states):
            probs, state_value, _ = self.model(
                state.unsqueeze(0),
                temperature=self.temperature,
                step=self.global_step - len(states) + i
            )
            
            cw_dist = torch.distributions.Categorical(probs[0])
            ccw_dist = torch.distributions.Categorical(probs[1])
            
            log_prob = cw_dist.log_prob(torch.tensor([actions[i][0]])) + \
                       ccw_dist.log_prob(torch.tensor([actions[i][1]]))
            policy_loss -= (log_prob * advantages[i]).mean()
            
            entropy = -(cw_dist.probs * cw_dist.logits).sum() - \
                      (ccw_dist.probs * ccw_dist.logits).sum()
            entropy_loss -= entropy * 0.01
            
            # ✅ 乾淨的維度匹配
            value_loss += nn.functional.mse_loss(state_value, returns[i:i+1])
        
        loss = policy_loss + 0.5 * value_loss + entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        return episode_reward, loss.item(), meta_info[0], diversity_score, avg_phase_diff

# ===== 主程序 =====
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, 
                       choices=['symmetry', 'explore', 'bifurcation'],
                       default='explore',
                       help='Evolution mode')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--learnable-phase', action='store_true',
                       help='Make phase velocity learnable')
    args = parser.parse_args()
    
    mode_map = {
        'symmetry': EvolutionMode.SYMMETRY_SEEKING,
        'explore': EvolutionMode.LONG_EXPLORATION,
        'bifurcation': EvolutionMode.PHASE_BIFURCATION
    }
    mode = mode_map[args.mode]
    
    print("="*70)
    print(f"  高級雙螺旋系統 - {mode.value}")
    print("="*70)
    print(f"\n模式說明:")
    if mode == EvolutionMode.SYMMETRY_SEEKING:
        print("  目標: 尋找並維持正逆螺旋的對稱解")
        print("  策略: 高對稱獎勵，低溫收斂")
    elif mode == EvolutionMode.LONG_EXPLORATION:
        print("  目標: 持續破對稱探索，避免過早收斂")
        print("  策略: 高溫維持，熵地板保護")
    else:
        print("  目標: 研究相位分岔與動力學演化")
        print("  策略: 超高溫，相位噪聲注入")
    print()
    
    model = TunableCounterRotatingNetwork(
        learnable_phase_velocity=args.learnable_phase
    )
    trainer = AdvancedA2CTrainer(model, mode=mode)
    env = ConfigurableHelixEnv(mode=mode)
    
    rewards_hist = deque(maxlen=100)
    diversity_hist = deque(maxlen=100)
    
    for ep in range(args.episodes):
        r, loss, meta, div, phase = trainer.train_episode(env)
        rewards_hist.append(r)
        diversity_hist.append(div)
        
        if ep % 50 == 0:
            print(f"Ep {ep:4d} | R: {r:7.2f} | "
                  f"AvgR: {np.mean(rewards_hist):7.2f} | "
                  f"Div: {div:.3f} | "
                  f"Δφ: {phase:.3f} | "
                  f"T: {trainer.temperature:.3f}")
    
    # 測試
    print("\n" + "="*70)
    print("  測試結果")
    print("="*70)
    
    test_env = ConfigurableHelixEnv(max_steps=50, mode=mode)
    state = test_env.reset()
    
    for step in range(50):
        with torch.no_grad():
            probs, _, meta = model(state.unsqueeze(0), 
                                  temperature=0.5, 
                                  step=step)
        
        cw_dist = torch.distributions.Categorical(probs[0])
        ccw_dist = torch.distributions.Categorical(probs[1])
        cw = cw_dist.sample().item()
        ccw = ccw_dist.sample().item()
        
        state, _, done = test_env.step(cw, ccw)
        if done:
            break
    
    unique_pairs = len(set(zip(test_env.cw_history, test_env.ccw_history)))
    symmetry = sum(1 for i in range(len(test_env.cw_history))
                   if test_env.cw_history[i] + test_env.ccw_history[i] == 65)
    
    print(f"\n唯一狀態對: {unique_pairs}/{len(test_env.cw_history)} "
          f"({100*unique_pairs/len(test_env.cw_history):.1f}%)")
    print(f"對稱步數:   {symmetry}/{len(test_env.cw_history)} "
          f"({100*symmetry/len(test_env.cw_history):.1f}%)")
    
    torch.save(model.state_dict(), f'helix_{args.mode}.pt')
    print(f"\n✓ 模型已保存")

if __name__ == "__main__":
    import random
    random.seed(42)
    torch.manual_seed(42)
    main()
