# -*- coding: utf-8 -*-
# live_phase_demo.py
# 實時相位演化動態演示

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import math
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
from collections import deque

from bifurcation_enhanced import BifurcationNetwork, BifurcationEnv

class LivePhaseDemo:
    """實時相位演化演示器"""
    
    def __init__(self, model, env, max_history=200):
        self.model = model
        self.env = env
        self.max_history = max_history
        
        # 數據緩衝區
        self.phase_buffer = deque(maxlen=max_history)
        self.cw_buffer = deque(maxlen=max_history)
        self.ccw_buffer = deque(maxlen=max_history)
        self.velocity_buffer = deque(maxlen=max_history)
        self.reward_buffer = deque(maxlen=max_history)
        
        self.step_count = 0
        self.state = self.env.reset()
        
    def step(self):
        """執行一步並更新緩衝區"""
        with torch.no_grad():
            probs, _, meta = self.model(
                self.state.unsqueeze(0),
                temperature=1.5,
                step=self.step_count,
                noise_scale=0.2
            )
        
        cw_dist = torch.distributions.Categorical(probs[0])
        ccw_dist = torch.distributions.Categorical(probs[1])
        cw_action = cw_dist.sample().item()
        ccw_action = ccw_dist.sample().item()
        
        next_state, reward, done = self.env.step(cw_action, ccw_action)
        
        phase_diff = abs(meta['cw_phase'] - meta['ccw_phase'])
        
        self.phase_buffer.append(phase_diff)
        self.cw_buffer.append(cw_action + 1)
        self.ccw_buffer.append(ccw_action + 1)
        self.velocity_buffer.append(meta['phase_velocity'])
        self.reward_buffer.append(reward)
        
        self.state = next_state
        self.step_count += 1
        
        if done:
            self.state = self.env.reset()
        
        return meta
    
    def create_animation(self, num_frames=500, interval=50, save_path='phase_animation.gif'):
        """創建動畫"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)
        
        # 1. 相位差時間序列
        ax1 = fig.add_subplot(gs[0, :])
        line1, = ax1.plot([], [], 'b-', linewidth=2, label='Δφ')
        ax1.axhline(y=math.pi, color='r', linestyle='--', alpha=0.5, label='π')
        ax1.set_xlim(0, self.max_history)
        ax1.set_ylim(0, 2*math.pi)
        ax1.set_ylabel('Phase Difference Δφ (rad)', fontsize=11)
        ax1.set_title('Real-Time Phase Difference Evolution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 極坐標相位表示
        ax2 = fig.add_subplot(gs[1, 0], projection='polar')
        line2, = ax2.plot([], [], 'ro', markersize=10)
        ax2.set_title('Phase Angle (Polar)', fontsize=12)
        
        # 3. 狀態空間軌跡
        ax3 = fig.add_subplot(gs[1, 1])
        scatter3 = ax3.scatter([], [], c=[], cmap='viridis', s=20, alpha=0.6)
        ax3.plot([1, 64], [64, 1], 'r--', alpha=0.3)
        ax3.set_xlim(1, 64)
        ax3.set_ylim(1, 64)
        ax3.set_xlabel('CW State', fontsize=10)
        ax3.set_ylabel('CCW State', fontsize=10)
        ax3.set_title('State Space Trajectory', fontsize=12)
        
        # 4. 相位速度
        ax4 = fig.add_subplot(gs[1, 2])
        line4, = ax4.plot([], [], 'g-', linewidth=2)
        ax4.set_xlim(0, self.max_history)
        ax4.set_ylim(0, 2)
        ax4.set_ylabel('Phase Velocity', fontsize=10)
        ax4.set_title('Phase Velocity Dynamics', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # 5. 獎勵曲線
        ax5 = fig.add_subplot(gs[2, :2])
        line5, = ax5.plot([], [], 'orange', linewidth=1.5)
        ax5.set_xlim(0, self.max_history)
        ax5.set_ylim(-10, 20)
        ax5.set_xlabel('Time Step', fontsize=10)
        ax5.set_ylabel('Reward', fontsize=10)
        ax5.set_title('Reward Signal', fontsize=12)
        ax5.grid(True, alpha=0.3)
        
        # 6. 相位直方圖
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.set_xlabel('Δφ (rad)', fontsize=10)
        ax6.set_ylabel('Frequency', fontsize=10)
        ax6.set_title('Phase Difference Distribution', fontsize=12)
        
        # 文字信息
        info_text = fig.text(0.02, 0.02, '', fontsize=10, family='monospace',
                            verticalalignment='bottom')
        
        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            line4.set_data([], [])
            line5.set_data([], [])
            scatter3.set_offsets(np.empty((0, 2)))
            return line1, line2, line4, line5, scatter3, info_text
        
        def update(frame):
            # 執行一步
            meta = self.step()
            
            # 更新時間序列
            x = np.arange(len(self.phase_buffer))
            line1.set_data(x, list(self.phase_buffer))
            
            # 更新極坐標
            if len(self.phase_buffer) > 0:
                theta = [self.phase_buffer[-1]]
                r = [1.0]
                line2.set_data(theta, r)
            
            # 更新狀態軌跡
            if len(self.cw_buffer) > 0:
                points = np.array([[cw, ccw] for cw, ccw in 
                                  zip(self.cw_buffer, self.ccw_buffer)])
                scatter3.set_offsets(points)
                scatter3.set_array(np.arange(len(points)))
            
            # 更新速度
            line4.set_data(x, list(self.velocity_buffer))
            
            # 更新獎勵
            line5.set_data(x, list(self.reward_buffer))
            
            # 更新直方圖
            ax6.clear()
            if len(self.phase_buffer) > 10:
                ax6.hist(list(self.phase_buffer), bins=20, alpha=0.7, color='purple')
                ax6.axvline(x=math.pi, color='r', linestyle='--', linewidth=2)
            ax6.set_xlabel('Δφ (rad)', fontsize=10)
            ax6.set_ylabel('Frequency', fontsize=10)
            ax6.set_title('Phase Distribution', fontsize=12)
            
            # 更新信息文字
            info_str = f"Step: {self.step_count:4d} | "
            info_str += f"Δφ: {self.phase_buffer[-1]:.3f} rad | "
            info_str += f"Velocity: {meta['phase_velocity']:.3f} | "
            info_str += f"Amplitude: {meta['phase_amplitude']:.3f}"
            info_text.set_text(info_str)
            
            return line1, line2, line4, line5, scatter3, info_text
        
        anim = FuncAnimation(fig, update, init_func=init,
                           frames=num_frames, interval=interval, blit=False)
        
        if save_path:
            print(f"Saving animation to {save_path}（this may take a few minutes）...")
            anim.save(save_path, writer='pillow', fps=20, dpi=100)
            print(f"✓ Animation successfully saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        return anim

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=300, help='動畫幀數')
    parser.add_argument('--save', type=str, default='phase_animation_v2.gif', 
                       help='保存路徑（不指定則實時顯示）')
    parser.add_argument('--no-save', action='store_true', help='不保存，直接顯示')
    args = parser.parse_args()
    
    print("="*70)
    print("  實時相位演化動態演示")
    print("="*70)
    
    # 載入模型
    model = BifurcationNetwork()
    try:
        model.load_state_dict(torch.load('bifurcation_enhanced_v2.pt'))
        print("\n✓ 已載入訓練好的模型")
    except FileNotFoundError:
        print("\n⚠ 使用隨機初始化模型")
    
    # 創建演示
    env = BifurcationEnv(max_steps=1000)
    demo = LivePhaseDemo(model, env, max_history=200)
    
    save_path = None if args.no_save else args.save
    
    print(f"\n生成 {args.frames} 幀動畫...")
    if save_path:
        print("（這可能需要幾分鐘，請耐心等待）")
    
    demo.create_animation(num_frames=args.frames, interval=50, save_path=save_path)
    
    print("\n✓ 完成！")

if __name__ == "__main__":
    main()
