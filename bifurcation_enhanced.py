# -*- coding: utf-8 -*-
# bifurcation_enhanced.py
# 增強版相位分岔系統 - 修復溫度貼底與相位鎖死

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from collections import deque

class BifurcationNetwork(nn.Module):
    """專為分岔研究優化的網絡"""
    def __init__(self, feature_dim=32, hidden_dim=64, action_dim=64):
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
        
        # 🔧 可學習的相位參數（幅度更大）
        self.cw_phase_base = nn.Parameter(torch.tensor(0.0))
        self.ccw_phase_base = nn.Parameter(torch.tensor(math.pi))
        self.phase_velocity = nn.Parameter(torch.tensor(0.5))  # 提高初始值
        self.phase_amplitude = nn.Parameter(torch.tensor(1.0))  # 新增振幅參數
    
    def forward(self, state, temperature=1.0, step=0, noise_scale=0.1):
        t = (step % 200) / 200.0
        
        # 🔧 增強的相位動力學
        phase_oscillation = self.phase_amplitude * (
            torch.sin(2 * math.pi * t * self.phase_velocity) +
            0.5 * torch.cos(4 * math.pi * t * self.phase_velocity)
        )
        
        # 🔧 更強的相位噪聲
        if noise_scale > 0:
            phase_noise = torch.randn(1) * noise_scale
        else:
            phase_noise = 0
        
        cw_phase = self.cw_phase_base + phase_oscillation + phase_noise
        ccw_phase = self.ccw_phase_base - phase_oscillation - phase_noise
        
        cw_features = self.cw_encoder(state)
        ccw_features = self.ccw_encoder(state)
        
        cw_logits = self.cw_actor(cw_features)
        ccw_logits = self.ccw_actor(ccw_features)
        
        # 相位調製
        cw_rotation = torch.cos(cw_phase) * 0.5 + 0.5
        ccw_rotation = torch.sin(ccw_phase) * 0.5 + 0.5
        
        entropy_bonus = temperature * 0.15  # 提高熵獎勵
        
        cw_probs = torch.softmax((cw_logits + entropy_bonus) / temperature, dim=-1)
        ccw_probs = torch.softmax((ccw_logits + entropy_bonus) / temperature, dim=-1)
        
        coupling_strength = torch.sigmoid(torch.tensor(1.0 - temperature))
        
        mixed_cw = cw_rotation * cw_probs + (1 - cw_rotation) * ccw_probs
        mixed_ccw = ccw_rotation * ccw_probs + (1 - ccw_rotation) * cw_probs
        
        mixed_probs = torch.stack([mixed_cw, mixed_ccw])
        
        fused = torch.cat([cw_features, ccw_features], dim=-1)
        state_value = self.value_fusion(fused)
        
        return mixed_probs, state_value, {
            'cw_phase': cw_phase.item(),
            'ccw_phase': ccw_phase.item(),
            'phase_velocity': self.phase_velocity.item(),
            'phase_amplitude': self.phase_amplitude.item(),
            'coupling': coupling_strength.item()
        }

class BifurcationEnv:
    def __init__(self, max_steps=100):
        self.max_steps = max_steps
        self.reset()
        # 🔧 記錄相位歷史用於分析
        self.phase_history = []
    
    def reset(self):
        self.cw_state = np.random.randint(1, 65)
        self.ccw_state = np.random.randint(1, 65)
        self.cw_history = [self.cw_state]
        self.ccw_history = [self.ccw_state]
        self.step_count = 0
        self.stagnation_counter = 0
        self.phase_history = []
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
        
        # 🔧 添加相位變化率特徵
        if len(self.phase_history) > 0:
            features[16] = self.phase_history[-1] / (2 * math.pi)
        
        return torch.FloatTensor(features)
    
    def step(self, cw_action, ccw_action):
        old_cw = self.cw_state
        old_ccw = self.ccw_state
        
        self.cw_state = cw_action + 1
        self.ccw_state = ccw_action + 1
        self.cw_history.append(self.cw_state)
        self.ccw_history.append(self.ccw_state)
        self.step_count += 1
        
        # 記錄相位
        cw_angle = 2 * math.pi * self.cw_state / 64
        ccw_angle = 2 * math.pi * self.ccw_state / 64
        phase_diff = abs(cw_angle - ccw_angle)
        self.phase_history.append(phase_diff)
        
        if self.cw_state == old_cw and self.ccw_state == old_ccw:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
        
        reward = self._compute_reward(old_cw, old_ccw, phase_diff)
        done = self.step_count >= self.max_steps
        
        return self._get_state(), reward, done
    
    def _compute_reward(self, old_cw, old_ccw, phase_diff):
        reward = 0.0
        
        # 基礎探索獎勵
        is_new_pair = (self.cw_state, self.ccw_state) not in \
                      zip(self.cw_history[:-1], self.ccw_history[:-1])
        if is_new_pair:
            reward += 3.0
        
        if abs(self.cw_state - old_cw) > 0 or abs(self.ccw_state - old_ccw) > 0:
            reward += 1.0
        
        # 停滯懲罰
        if self.stagnation_counter >= 5:
            reward -= 5.0 * (self.stagnation_counter - 4)
        
        # 🔧 改進的相位獎勵（更寬鬆的區間）
        phase_targets = [0, math.pi/4, math.pi/2, 3*math.pi/4, 
                        math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]
        
        min_distance = min(abs(phase_diff - t) for t in phase_targets)
        
        if min_distance < 0.5:  # 擴大接受範圍
            reward += 4.0 * (1.0 - min_distance / 0.5)  # 漸變獎勵
        
        # 🔧 相位跳躍獎勵（鼓勵分岔）
        if len(self.phase_history) >= 2:
            phase_change = abs(self.phase_history[-1] - self.phase_history[-2])
            if phase_change > math.pi / 3:  # 大於 60 度
                reward += 3.0
            elif phase_change > math.pi / 6:  # 大於 30 度
                reward += 1.5
        
        # 🔧 懲罰長期相位鎖定
        if len(self.phase_history) >= 20:
            recent_phases = self.phase_history[-20:]
            phase_variance = np.var(recent_phases)
            if phase_variance < 0.1:  # 相位幾乎不變
                reward -= 2.0
        
        # 多樣性獎勵
        if len(self.cw_history) >= 10:
            recent_unique = len(set(self.cw_history[-10:]))
            reward += (recent_unique / 10.0) * 2.0
        
        return reward

class BifurcationTrainer:
    def __init__(self, model, lr=1e-3, gamma=0.99):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.global_step = 0
        
        # 🔧 更激進的溫度配置
        self.temperature = 3.0
        self.min_temperature = 1.5  # 提高下限
        self.max_temperature = 5.0  # 提高上限
        self.temperature_momentum = 0.0
        
        # 🔧 動態噪聲強度
        self.noise_scale = 0.5  # 顯著提高
    
    def adjust_temperature(self, diversity_score, phase_variance):
        """根據多樣性和相位方差調整溫度"""
        
        # 如果相位方差太小（鎖定），強制升溫
        if phase_variance < 0.2:
            self.temperature_momentum += 0.15
        elif phase_variance > 0.8:
            self.temperature_momentum -= 0.05
        
        # 多樣性檢查
        if diversity_score < 0.7:
            self.temperature_momentum += 0.1
        elif diversity_score > 0.98:
            self.temperature_momentum -= 0.03
        
        # 🔧 防止貼底機制
        if self.temperature < self.min_temperature + 0.5:
            self.temperature_momentum += 0.1
        
        self.temperature += self.temperature_momentum
        self.temperature = np.clip(
            self.temperature, 
            self.min_temperature, 
            self.max_temperature
        )
        self.temperature_momentum *= 0.85
        
        # 🔧 動態調整噪聲
        self.noise_scale = 0.1 + 0.3 * (self.temperature - self.min_temperature) / \
                          (self.max_temperature - self.min_temperature)
    
    def train_episode(self, env):
        state = env.reset()
        states, actions, rewards, values = [], [], [], []
        episode_reward = 0
        meta_info = []
        
        done = False
        while not done:
            with torch.no_grad():
                probs, state_value, info = self.model(
                    state.unsqueeze(0),
                    temperature=self.temperature,
                    step=self.global_step,
                    noise_scale=self.noise_scale
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
        
        # 計算相位方差
        phase_variance = np.var(env.phase_history) if len(env.phase_history) > 0 else 0
        avg_phase = np.mean([abs(m['cw_phase'] - m['ccw_phase']) 
                            for m in meta_info])
        
        self.adjust_temperature(diversity_score, phase_variance)
        
        # 訓練
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
                step=self.global_step - len(states) + i,
                noise_scale=0  # 訓練時不加噪聲
            )
            
            cw_dist = torch.distributions.Categorical(probs[0])
            ccw_dist = torch.distributions.Categorical(probs[1])
            
            log_prob = cw_dist.log_prob(torch.tensor([actions[i][0]])) + \
                       ccw_dist.log_prob(torch.tensor([actions[i][1]]))
            policy_loss -= (log_prob * advantages[i]).mean()
            
            entropy = -(cw_dist.probs * cw_dist.logits).sum() - \
                      (ccw_dist.probs * ccw_dist.logits).sum()
            entropy_loss -= entropy * 0.02  # 提高熵權重
            
            value_loss += nn.functional.mse_loss(state_value, returns[i:i+1])
        
        loss = policy_loss + 0.5 * value_loss + entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        return episode_reward, loss.item(), meta_info[0], diversity_score, \
               phase_variance, avg_phase

def main():
    print("="*70)
    print("  增強版相位分岔研究系統")
    print("="*70)
    print("\n改進:")
    print("  ✓ 提高溫度下限 (1.0 → 1.5)")
    print("  ✓ 增強相位噪聲 (0.01 → 0.3)")
    print("  ✓ 添加相位方差監控")
    print("  ✓ 改進相位獎勵函數")
    print("  ✓ 防止溫度貼底機制\n")
    
    model = BifurcationNetwork()
    trainer = BifurcationTrainer(model, lr=1e-3)
    env = BifurcationEnv()
    
    rewards_hist = deque(maxlen=100)
    diversity_hist = deque(maxlen=100)
    phase_var_hist = deque(maxlen=100)
    
    for ep in range(500):
        r, loss, meta, div, phase_var, avg_phase = trainer.train_episode(env)
        rewards_hist.append(r)
        diversity_hist.append(div)
        phase_var_hist.append(phase_var)
        
        if ep % 50 == 0:
            print(f"Ep {ep:4d} | "
                  f"R: {r:7.2f} | "
                  f"AvgR: {np.mean(rewards_hist):7.2f} | "
                  f"Div: {div:.3f} | "
                  f"φVar: {phase_var:.3f} | "
                  f"Δφ: {avg_phase:.3f} | "
                  f"T: {trainer.temperature:.3f} | "
                  f"Noise: {trainer.noise_scale:.3f}")
    
    # 詳細測試
    print("\n" + "="*70)
    print("  相位分岔測試")
    print("="*70)
    
    test_env = BifurcationEnv(max_steps=100)
    state = test_env.reset()
    
    print(f"\n{'Step':>4} | {'CW':>3} | {'CCW':>3} | {'Δφ(rad)':>8} | {'φVel':>6}")
    print("-" * 60)
    
    for step in range(100):
        with torch.no_grad():
            probs, _, meta = model(
                state.unsqueeze(0), 
                temperature=1.5, 
                step=step,
                noise_scale=0.2
            )
        
        cw_dist = torch.distributions.Categorical(probs[0])
        ccw_dist = torch.distributions.Categorical(probs[1])
        cw = cw_dist.sample().item()
        ccw = ccw_dist.sample().item()
        
        state, _, done = test_env.step(cw, ccw)
        
        if step % 10 == 0:
            phase_diff = abs(meta['cw_phase'] - meta['ccw_phase'])
            print(f"{step:4d} | {cw+1:3d} | {ccw+1:3d} | "
                  f"{phase_diff:8.3f} | {meta['phase_velocity']:6.3f}")
        
        if done:
            break
    
    # 統計
    unique_pairs = len(set(zip(test_env.cw_history, test_env.ccw_history)))
    phase_var = np.var(test_env.phase_history)
    phase_mean = np.mean(test_env.phase_history)
    
    print(f"\n測試統計:")
    print(f"  唯一狀態對:   {unique_pairs}/{len(test_env.cw_history)} "
          f"({100*unique_pairs/len(test_env.cw_history):.1f}%)")
    print(f"  相位方差:     {phase_var:.4f}")
    print(f"  平均相位差:   {phase_mean:.4f} rad ({phase_mean*180/math.pi:.1f}°)")
    print(f"  學習到的參數:")
    print(f"    phase_velocity:  {model.phase_velocity.item():.4f}")
    print(f"    phase_amplitude: {model.phase_amplitude.item():.4f}")
    
    torch.save(model.state_dict(), 'bifurcation_enhanced.pt')
    print(f"\n✓ 模型已保存")

if __name__ == "__main__":
    import random
    random.seed(42)
    torch.manual_seed(42)
    main()
