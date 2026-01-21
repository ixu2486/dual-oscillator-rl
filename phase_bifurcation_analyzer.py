# -*- coding: utf-8 -*-
# phase_bifurcation_analyzer.py
# 相位分岔與動力學可視化分析工具

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import seaborn as sns
from collections import deque
import math

# 設置中文字體和樣式
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("darkgrid")

# 從 bifurcation_enhanced 導入模型
from bifurcation_enhanced import BifurcationNetwork, BifurcationEnv, BifurcationTrainer

class PhaseAnalyzer:
    """相位動力學分析器"""
    
    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.phase_history = []
        self.cw_history = []
        self.ccw_history = []
        self.velocity_history = []
        self.amplitude_history = []
        self.temperature_history = []
        self.reward_history = []
        self.diversity_history = []
        
    def collect_trajectory(self, num_steps=500, temperature=1.5, noise_scale=0.2):
        """收集一條完整軌跡"""
        state = self.env.reset()
        
        for step in range(num_steps):
            with torch.no_grad():
                probs, _, meta = self.model(
                    state.unsqueeze(0),
                    temperature=temperature,
                    step=step,
                    noise_scale=noise_scale
                )
            
            cw_dist = torch.distributions.Categorical(probs[0])
            ccw_dist = torch.distributions.Categorical(probs[1])
            cw_action = cw_dist.sample().item()
            ccw_action = ccw_dist.sample().item()
            
            next_state, reward, done = self.env.step(cw_action, ccw_action)
            
            # 記錄數據
            phase_diff = abs(meta['cw_phase'] - meta['ccw_phase'])
            self.phase_history.append(phase_diff)
            self.cw_history.append(cw_action + 1)
            self.ccw_history.append(ccw_action + 1)
            self.velocity_history.append(meta['phase_velocity'])
            self.amplitude_history.append(meta['phase_amplitude'])
            self.reward_history.append(reward)
            
            # 計算當前多樣性
            unique_pairs = len(set(zip(self.env.cw_history, self.env.ccw_history)))
            diversity = unique_pairs / len(self.env.cw_history)
            self.diversity_history.append(diversity)
            
            state = next_state
            if done:
                break
    
    def plot_phase_bifurcation(self, save_path='phase_bifurcation.png'):
        """繪製相位分岔圖"""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 相位差時間序列
        ax1 = fig.add_subplot(gs[0, :])
        steps = np.arange(len(self.phase_history))
        ax1.plot(steps, self.phase_history, linewidth=1.5, alpha=0.7, color='#2E86AB')
        ax1.axhline(y=math.pi, color='r', linestyle='--', alpha=0.5, label='π (反相)')
        ax1.axhline(y=math.pi/2, color='g', linestyle='--', alpha=0.5, label='π/2 (正交)')
        ax1.axhline(y=0, color='b', linestyle='--', alpha=0.5, label='0 (同相)')
        ax1.set_xlabel('時間步 (Step)', fontsize=12)
        ax1.set_ylabel('相位差 Δφ (rad)', fontsize=12)
        ax1.set_title('相位差動態演化 (Phase Difference Evolution)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 相位空間軌跡 (CW vs CCW)
        ax2 = fig.add_subplot(gs[1, 0])
        scatter = ax2.scatter(self.cw_history, self.ccw_history, 
                             c=self.phase_history, cmap='viridis', 
                             s=10, alpha=0.6)
        ax2.plot([1, 64], [64, 1], 'r--', alpha=0.5, label='完美對稱線')
        ax2.set_xlabel('正螺旋狀態 (CW State)', fontsize=11)
        ax2.set_ylabel('逆螺旋狀態 (CCW State)', fontsize=11)
        ax2.set_title('狀態空間軌跡 (State Space Trajectory)', fontsize=13, fontweight='bold')
        ax2.legend()
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Δφ (rad)', fontsize=10)
        
        # 3. 相位分佈直方圖
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(self.phase_history, bins=50, alpha=0.7, color='#A23B72', edgecolor='black')
        ax3.axvline(x=math.pi, color='r', linestyle='--', linewidth=2, label='π')
        ax3.axvline(x=math.pi/2, color='g', linestyle='--', linewidth=2, label='π/2')
        ax3.set_xlabel('相位差 Δφ (rad)', fontsize=11)
        ax3.set_ylabel('頻次 (Frequency)', fontsize=11)
        ax3.set_title('相位分佈 (Phase Distribution)', fontsize=13, fontweight='bold')
        ax3.legend()
        
        # 4. 相位速度演化
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(steps, self.velocity_history, linewidth=1.5, color='#F18F01')
        ax4.set_xlabel('時間步 (Step)', fontsize=11)
        ax4.set_ylabel('相位速度 (Phase Velocity)', fontsize=11)
        ax4.set_title('相位速度演化 (Phase Velocity Evolution)', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. 相位振幅演化
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(steps, self.amplitude_history, linewidth=1.5, color='#C73E1D')
        ax5.set_xlabel('時間步 (Step)', fontsize=11)
        ax5.set_ylabel('相位振幅 (Phase Amplitude)', fontsize=11)
        ax5.set_title('相位振幅演化 (Phase Amplitude Evolution)', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 相位分岔圖已保存至 {save_path}")
        plt.close()
    
    def plot_ripple_dynamics(self, save_path='ripple_dynamics.png'):
        """繪製漣漪效應 - Δφ 隨時間的動態"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        steps = np.arange(len(self.phase_history))
        
        # 1. Δφ 漣漪圖（帶移動平均）
        window = 20
        phase_ma = np.convolve(self.phase_history, np.ones(window)/window, mode='same')
        
        axes[0].fill_between(steps, self.phase_history, alpha=0.3, color='#2E86AB', label='原始 Δφ')
        axes[0].plot(steps, phase_ma, linewidth=2, color='#A23B72', label=f'移動平均 ({window} 步)')
        axes[0].axhline(y=math.pi, color='r', linestyle='--', alpha=0.5)
        axes[0].set_ylabel('Δφ (rad)', fontsize=12)
        axes[0].set_title('相位差漣漪效應 (Phase Ripple Effect)', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 相位變化率 (dΔφ/dt)
        phase_derivative = np.diff(self.phase_history, prepend=self.phase_history[0])
        axes[1].plot(steps, phase_derivative, linewidth=1, alpha=0.7, color='#F18F01')
        axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1].set_ylabel('dΔφ/dt', fontsize=12)
        axes[1].set_title('相位變化率 (Phase Change Rate)', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 頻譜分析 (FFT)
        fft_result = np.fft.fft(self.phase_history - np.mean(self.phase_history))
        freqs = np.fft.fftfreq(len(self.phase_history))
        power = np.abs(fft_result[:len(fft_result)//2])**2
        freqs = freqs[:len(freqs)//2]
        
        axes[2].semilogy(freqs[1:], power[1:], linewidth=1.5, color='#C73E1D')
        axes[2].set_xlabel('頻率 (Frequency)', fontsize=12)
        axes[2].set_ylabel('功率譜密度 (Power)', fontsize=12)
        axes[2].set_title('相位振盪頻譜 (Phase Oscillation Spectrum)', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 漣漪動態圖已保存至 {save_path}")
        plt.close()
    
    def plot_parameter_impact(self, save_path='parameter_impact.png'):
        """分析 phase_velocity 和 amplitude 對探索的影響"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        steps = np.arange(len(self.phase_history))
        
        # 1. Velocity vs Diversity
        axes[0, 0].scatter(self.velocity_history, self.diversity_history, 
                          c=steps, cmap='plasma', s=10, alpha=0.6)
        axes[0, 0].set_xlabel('Phase Velocity', fontsize=11)
        axes[0, 0].set_ylabel('Diversity', fontsize=11)
        axes[0, 0].set_title('速度-多樣性關係 (Velocity vs Diversity)', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Amplitude vs Phase Variance
        window = 50
        phase_variance = []
        for i in range(len(self.phase_history)):
            start = max(0, i - window)
            phase_variance.append(np.var(self.phase_history[start:i+1]))
        
        axes[0, 1].scatter(self.amplitude_history, phase_variance,
                          c=steps, cmap='plasma', s=10, alpha=0.6)
        axes[0, 1].set_xlabel('Phase Amplitude', fontsize=11)
        axes[0, 1].set_ylabel('Phase Variance (50-step window)', fontsize=11)
        axes[0, 1].set_title('振幅-相位方差關係 (Amplitude vs Phase Variance)', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Velocity 時間序列 + Reward
        ax3a = axes[1, 0]
        ax3b = ax3a.twinx()
        
        line1 = ax3a.plot(steps, self.velocity_history, color='#2E86AB', 
                         linewidth=1.5, label='Velocity')
        line2 = ax3b.plot(steps, self.reward_history, color='#F18F01', 
                         linewidth=1, alpha=0.7, label='Reward')
        
        ax3a.set_xlabel('時間步 (Step)', fontsize=11)
        ax3a.set_ylabel('Phase Velocity', fontsize=11, color='#2E86AB')
        ax3b.set_ylabel('Reward', fontsize=11, color='#F18F01')
        ax3a.set_title('速度與獎勵共演化 (Velocity-Reward Co-evolution)', fontsize=12, fontweight='bold')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3a.legend(lines, labels, loc='upper left')
        ax3a.grid(True, alpha=0.3)
        
        # 4. Amplitude 時間序列 + Diversity
        ax4a = axes[1, 1]
        ax4b = ax4a.twinx()
        
        line1 = ax4a.plot(steps, self.amplitude_history, color='#C73E1D',
                         linewidth=1.5, label='Amplitude')
        line2 = ax4b.plot(steps, self.diversity_history, color='#00A878',
                         linewidth=1, alpha=0.7, label='Diversity')
        
        ax4a.set_xlabel('時間步 (Step)', fontsize=11)
        ax4a.set_ylabel('Phase Amplitude', fontsize=11, color='#C73E1D')
        ax4b.set_ylabel('Diversity', fontsize=11, color='#00A878')
        ax4a.set_title('振幅與多樣性共演化 (Amplitude-Diversity Co-evolution)', fontsize=12, fontweight='bold')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4a.legend(lines, labels, loc='upper left')
        ax4a.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 參數影響分析圖已保存至 {save_path}")
        plt.close()
    
    def plot_phase_portrait(self, save_path='phase_portrait.png'):
        """繪製相位肖像（吸引子分析）"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. 2D 相位空間 (Δφ vs dΔφ/dt)
        phase_derivative = np.diff(self.phase_history, prepend=self.phase_history[0])
        
        axes[0, 0].scatter(self.phase_history, phase_derivative,
                          c=np.arange(len(self.phase_history)), 
                          cmap='coolwarm', s=5, alpha=0.5)
        axes[0, 0].set_xlabel('Δφ', fontsize=11)
        axes[0, 0].set_ylabel('dΔφ/dt', fontsize=11)
        axes[0, 0].set_title('相位肖像 (Phase Portrait)', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Poincaré 截面 (當 dΔφ/dt > 0 時的 Δφ)
        poincare_points = [self.phase_history[i] for i in range(len(phase_derivative)) 
                           if phase_derivative[i] > 0 and i > 0 and phase_derivative[i-1] <= 0]
        
        if len(poincare_points) > 0:
            axes[0, 1].hist(poincare_points, bins=30, alpha=0.7, color='#A23B72', edgecolor='black')
            axes[0, 1].axvline(x=math.pi, color='r', linestyle='--', linewidth=2)
            axes[0, 1].set_xlabel('Δφ at crossing', fontsize=11)
            axes[0, 1].set_ylabel('Frequency', fontsize=11)
            axes[0, 1].set_title(f'Poincaré 截面 ({len(poincare_points)} 次穿越)', fontsize=12, fontweight='bold')
        
        # 3. 回歸圖 (Return Map): Δφ[n+1] vs Δφ[n]
        axes[1, 0].scatter(self.phase_history[:-1], self.phase_history[1:],
                          c=np.arange(len(self.phase_history)-1),
                          cmap='viridis', s=5, alpha=0.5)
        axes[1, 0].plot([0, 2*math.pi], [0, 2*math.pi], 'r--', alpha=0.5, label='對角線')
        axes[1, 0].set_xlabel('Δφ[n]', fontsize=11)
        axes[1, 0].set_ylabel('Δφ[n+1]', fontsize=11)
        axes[1, 0].set_title('回歸圖 (Return Map)', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 李雅普諾夫指數估算 (簡化版)
        def estimate_lyapunov(data, delay=10):
            divergences = []
            for i in range(len(data) - delay - 1):
                diff = abs(data[i+delay] - data[i])
                if diff > 0:
                    divergences.append(np.log(diff))
            return np.mean(divergences) if divergences else 0
        
        lyapunov_estimates = []
        window = 100
        for i in range(window, len(self.phase_history), 10):
            segment = self.phase_history[i-window:i]
            lyap = estimate_lyapunov(segment)
            lyapunov_estimates.append(lyap)
        
        if len(lyapunov_estimates) > 0:
            axes[1, 1].plot(lyapunov_estimates, linewidth=1.5, color='#F18F01')
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='穩定/混沌邊界')
            axes[1, 1].set_xlabel('時間窗口', fontsize=11)
            axes[1, 1].set_ylabel('李雅普諾夫指數估計', fontsize=11)
            axes[1, 1].set_title('混沌度量 (Chaos Indicator)', fontsize=12, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 相位肖像圖已保存至 {save_path}")
        plt.close()
    
    def generate_summary_report(self):
        """生成分析摘要報告"""
        print("\n" + "="*70)
        print("  相位動力學分析報告")
        print("="*70)
        
        # 基本統計
        print(f"\n基本統計:")
        print(f"  軌跡長度:        {len(self.phase_history)} 步")
        print(f"  平均相位差:      {np.mean(self.phase_history):.4f} rad ({np.mean(self.phase_history)*180/math.pi:.2f}°)")
        print(f"  相位差標準差:    {np.std(self.phase_history):.4f} rad")
        print(f"  相位差範圍:      [{np.min(self.phase_history):.4f}, {np.max(self.phase_history):.4f}]")
        
        # 參數統計
        print(f"\n學習參數:")
        print(f"  最終 phase_velocity:  {self.velocity_history[-1]:.4f}")
        print(f"  最終 phase_amplitude: {self.amplitude_history[-1]:.4f}")
        print(f"  velocity 變化範圍:    [{np.min(self.velocity_history):.4f}, {np.max(self.velocity_history):.4f}]")
        print(f"  amplitude 變化範圍:   [{np.min(self.amplitude_history):.4f}, {np.max(self.amplitude_history):.4f}]")
        
        # 探索統計
        print(f"\n探索性能:")
        print(f"  平均多樣性:      {np.mean(self.diversity_history):.4f}")
        print(f"  平均獎勵:        {np.mean(self.reward_history):.2f}")
        print(f"  唯一狀態對數:    {len(set(zip(self.cw_history, self.ccw_history)))}")
        
        # 相位吸引子檢測
        phase_bins = np.histogram(self.phase_history, bins=8, range=(0, 2*math.pi))[0]
        attractors = [i for i, count in enumerate(phase_bins) if count > len(self.phase_history) * 0.15]
        
        print(f"\n相位吸引子:")
        if attractors:
            for idx in attractors:
                phase_center = (idx + 0.5) * 2 * math.pi / 8
                percentage = phase_bins[idx] / len(self.phase_history) * 100
                print(f"  吸引子 {idx+1}: {phase_center:.4f} rad ({phase_center*180/math.pi:.1f}°) - {percentage:.1f}%")
        else:
            print("  未檢測到顯著吸引子（相位分佈均勻）")
        
        print("\n" + "="*70 + "\n")

def main():
    print("="*70)
    print("  相位分岔可視化分析工具")
    print("="*70)
    
    # 載入或創建模型
    model = BifurcationNetwork()
    
    try:
        model.load_state_dict(torch.load('bifurcation_enhanced_v2.pt'))
        print("\n✓ 已載入訓練好的模型")
    except FileNotFoundError:
        print("\n⚠ 未找到訓練模型，使用隨機初始化")
        print("  建議先運行 bifurcation_enhanced.py 進行訓練")
    
    # 創建環境和分析器
    env = BifurcationEnv(max_steps=1000)
    analyzer = PhaseAnalyzer(model, env)
    
    # 收集數據
    print("\n收集軌跡數據...")
    analyzer.collect_trajectory(num_steps=1000, temperature=1.5, noise_scale=0.2)
    
    # 生成所有可視化
    print("\n生成可視化圖表...")
    analyzer.plot_phase_bifurcation()
    analyzer.plot_ripple_dynamics()
    analyzer.plot_parameter_impact()
    analyzer.plot_phase_portrait()
    
    # 生成報告
    analyzer.generate_summary_report()
    
    print("✓ 所有分析完成！")
    print("\n生成的文件:")
    print("  - phase_bifurcation.png  (相位分岔圖)")
    print("  - ripple_dynamics.png    (漣漪動態)")
    print("  - parameter_impact.png   (參數影響)")
    print("  - phase_portrait.png     (相位肖像)")

if __name__ == "__main__":
    main()
