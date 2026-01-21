# -*- coding: utf-8 -*-
# phase_bifurcation_analyzer_en.py
# Phase bifurcation and dynamics visualization tool (English version)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import seaborn as sns
from collections import deque
import math

# Set style (English-compatible)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("darkgrid")

# Import model from bifurcation_enhanced
try:
    from bifurcation_enhanced_v2 import BifurcationNetwork, BifurcationEnv, BifurcationTrainer
except ImportError:
    print("Warning: Could not import from bifurcation_enhanced.py")
    print("Make sure the file exists in the same directory")

class PhaseAnalyzer:
    """Phase dynamics analyzer"""
    
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
        """Collect a complete trajectory"""
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
            
            # Record data
            phase_diff = abs(meta['cw_phase'] - meta['ccw_phase'])
            self.phase_history.append(phase_diff)
            self.cw_history.append(cw_action + 1)
            self.ccw_history.append(ccw_action + 1)
            self.velocity_history.append(meta['phase_velocity'])
            self.amplitude_history.append(meta['phase_amplitude'])
            self.reward_history.append(reward)
            
            # Calculate current diversity
            unique_pairs = len(set(zip(self.env.cw_history, self.env.ccw_history)))
            diversity = unique_pairs / len(self.env.cw_history)
            self.diversity_history.append(diversity)
            
            state = next_state
            if done:
                break
    
    def plot_phase_bifurcation(self, save_path='phase_bifurcation.png'):
        """Plot phase bifurcation diagram"""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Phase difference time series
        ax1 = fig.add_subplot(gs[0, :])
        steps = np.arange(len(self.phase_history))
        ax1.plot(steps, self.phase_history, linewidth=1.5, alpha=0.7, color='#2E86AB')
        ax1.axhline(y=math.pi, color='r', linestyle='--', alpha=0.5, label='π (anti-phase)')
        ax1.axhline(y=math.pi/2, color='g', linestyle='--', alpha=0.5, label='π/2 (orthogonal)')
        ax1.axhline(y=0, color='b', linestyle='--', alpha=0.5, label='0 (in-phase)')
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Phase Difference Δφ (rad)', fontsize=12)
        ax1.set_title('Phase Difference Evolution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. State space trajectory (CW vs CCW)
        ax2 = fig.add_subplot(gs[1, 0])
        scatter = ax2.scatter(self.cw_history, self.ccw_history, 
                             c=self.phase_history, cmap='viridis', 
                             s=10, alpha=0.6)
        ax2.plot([1, 64], [64, 1], 'r--', alpha=0.5, label='Perfect symmetry')
        ax2.set_xlabel('CW State', fontsize=11)
        ax2.set_ylabel('CCW State', fontsize=11)
        ax2.set_title('State Space Trajectory', fontsize=13, fontweight='bold')
        ax2.legend()
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Δφ (rad)', fontsize=10)
        
        # 3. Phase distribution histogram
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(self.phase_history, bins=50, alpha=0.7, color='#A23B72', edgecolor='black')
        ax3.axvline(x=math.pi, color='r', linestyle='--', linewidth=2, label='π')
        ax3.axvline(x=math.pi/2, color='g', linestyle='--', linewidth=2, label='π/2')
        ax3.set_xlabel('Phase Difference Δφ (rad)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Phase Distribution', fontsize=13, fontweight='bold')
        ax3.legend()
        
        # 4. Phase velocity evolution
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(steps, self.velocity_history, linewidth=1.5, color='#F18F01')
        ax4.set_xlabel('Time Step', fontsize=11)
        ax4.set_ylabel('Phase Velocity', fontsize=11)
        ax4.set_title('Phase Velocity Evolution', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Phase amplitude evolution
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(steps, self.amplitude_history, linewidth=1.5, color='#C73E1D')
        ax5.set_xlabel('Time Step', fontsize=11)
        ax5.set_ylabel('Phase Amplitude', fontsize=11)
        ax5.set_title('Phase Amplitude Evolution', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Phase bifurcation diagram saved to {save_path}")
        plt.close()
    
    def plot_ripple_dynamics(self, save_path='ripple_dynamics.png'):
        """Plot ripple effect - Δφ dynamics over time"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        steps = np.arange(len(self.phase_history))
        
        # 1. Δφ ripple plot (with moving average)
        window = 20
        phase_ma = np.convolve(self.phase_history, np.ones(window)/window, mode='same')
        
        axes[0].fill_between(steps, self.phase_history, alpha=0.3, color='#2E86AB', label='Raw Δφ')
        axes[0].plot(steps, phase_ma, linewidth=2, color='#A23B72', label=f'Moving Avg ({window} steps)')
        axes[0].axhline(y=math.pi, color='r', linestyle='--', alpha=0.5)
        axes[0].set_ylabel('Δφ (rad)', fontsize=12)
        axes[0].set_title('Phase Ripple Effect', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Phase change rate (dΔφ/dt)
        phase_derivative = np.diff(self.phase_history, prepend=self.phase_history[0])
        axes[1].plot(steps, phase_derivative, linewidth=1, alpha=0.7, color='#F18F01')
        axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1].set_ylabel('dΔφ/dt', fontsize=12)
        axes[1].set_title('Phase Change Rate', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Spectral analysis (FFT)
        fft_result = np.fft.fft(self.phase_history - np.mean(self.phase_history))
        freqs = np.fft.fftfreq(len(self.phase_history))
        power = np.abs(fft_result[:len(fft_result)//2])**2
        freqs = freqs[:len(freqs)//2]
        
        axes[2].semilogy(freqs[1:], power[1:], linewidth=1.5, color='#C73E1D')
        axes[2].set_xlabel('Frequency', fontsize=12)
        axes[2].set_ylabel('Power Spectral Density', fontsize=12)
        axes[2].set_title('Phase Oscillation Spectrum', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Ripple dynamics plot saved to {save_path}")
        plt.close()
    
    def plot_parameter_impact(self, save_path='parameter_impact.png'):
        """Analyze phase_velocity and amplitude impact on exploration"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        steps = np.arange(len(self.phase_history))
        
        # 1. Velocity vs Diversity
        axes[0, 0].scatter(self.velocity_history, self.diversity_history, 
                          c=steps, cmap='plasma', s=10, alpha=0.6)
        axes[0, 0].set_xlabel('Phase Velocity', fontsize=11)
        axes[0, 0].set_ylabel('Diversity', fontsize=11)
        axes[0, 0].set_title('Velocity vs Diversity', fontsize=12, fontweight='bold')
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
        axes[0, 1].set_title('Amplitude vs Phase Variance', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Velocity time series + Reward
        ax3a = axes[1, 0]
        ax3b = ax3a.twinx()
        
        line1 = ax3a.plot(steps, self.velocity_history, color='#2E86AB', 
                         linewidth=1.5, label='Velocity')
        line2 = ax3b.plot(steps, self.reward_history, color='#F18F01', 
                         linewidth=1, alpha=0.7, label='Reward')
        
        ax3a.set_xlabel('Time Step', fontsize=11)
        ax3a.set_ylabel('Phase Velocity', fontsize=11, color='#2E86AB')
        ax3b.set_ylabel('Reward', fontsize=11, color='#F18F01')
        ax3a.set_title('Velocity-Reward Co-evolution', fontsize=12, fontweight='bold')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3a.legend(lines, labels, loc='upper left')
        ax3a.grid(True, alpha=0.3)
        
        # 4. Amplitude time series + Diversity
        ax4a = axes[1, 1]
        ax4b = ax4a.twinx()
        
        line1 = ax4a.plot(steps, self.amplitude_history, color='#C73E1D',
                         linewidth=1.5, label='Amplitude')
        line2 = ax4b.plot(steps, self.diversity_history, color='#00A878',
                         linewidth=1, alpha=0.7, label='Diversity')
        
        ax4a.set_xlabel('Time Step', fontsize=11)
        ax4a.set_ylabel('Phase Amplitude', fontsize=11, color='#C73E1D')
        ax4b.set_ylabel('Diversity', fontsize=11, color='#00A878')
        ax4a.set_title('Amplitude-Diversity Co-evolution', fontsize=12, fontweight='bold')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4a.legend(lines, labels, loc='upper left')
        ax4a.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Parameter impact analysis saved to {save_path}")
        plt.close()
    
    def plot_phase_portrait(self, save_path='phase_portrait.png'):
        """Plot phase portrait (attractor analysis)"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. 2D phase space (Δφ vs dΔφ/dt)
        phase_derivative = np.diff(self.phase_history, prepend=self.phase_history[0])
        
        axes[0, 0].scatter(self.phase_history, phase_derivative,
                          c=np.arange(len(self.phase_history)), 
                          cmap='coolwarm', s=5, alpha=0.5)
        axes[0, 0].set_xlabel('Δφ', fontsize=11)
        axes[0, 0].set_ylabel('dΔφ/dt', fontsize=11)
        axes[0, 0].set_title('Phase Portrait', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Poincaré section (Δφ when dΔφ/dt crosses zero)
        poincare_points = [self.phase_history[i] for i in range(len(phase_derivative)) 
                           if phase_derivative[i] > 0 and i > 0 and phase_derivative[i-1] <= 0]
        
        if len(poincare_points) > 0:
            axes[0, 1].hist(poincare_points, bins=30, alpha=0.7, color='#A23B72', edgecolor='black')
            axes[0, 1].axvline(x=math.pi, color='r', linestyle='--', linewidth=2)
            axes[0, 1].set_xlabel('Δφ at crossing', fontsize=11)
            axes[0, 1].set_ylabel('Frequency', fontsize=11)
            axes[0, 1].set_title(f'Poincare Section ({len(poincare_points)} crossings)', fontsize=12, fontweight='bold')
        
        # 3. Return map: Δφ[n+1] vs Δφ[n]
        axes[1, 0].scatter(self.phase_history[:-1], self.phase_history[1:],
                          c=np.arange(len(self.phase_history)-1),
                          cmap='viridis', s=5, alpha=0.5)
        axes[1, 0].plot([0, 2*math.pi], [0, 2*math.pi], 'r--', alpha=0.5, label='Diagonal')
        axes[1, 0].set_xlabel('Δφ[n]', fontsize=11)
        axes[1, 0].set_ylabel('Δφ[n+1]', fontsize=11)
        axes[1, 0].set_title('Return Map', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Lyapunov exponent estimation (simplified)
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
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Stability/Chaos boundary')
            axes[1, 1].set_xlabel('Time Window', fontsize=11)
            axes[1, 1].set_ylabel('Lyapunov Exponent Estimate', fontsize=11)
            axes[1, 1].set_title('Chaos Indicator', fontsize=12, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Phase portrait saved to {save_path}")
        plt.close()
    
    def generate_summary_report(self):
        """Generate analysis summary report"""
        print("\n" + "="*70)
        print("  Phase Dynamics Analysis Report")
        print("="*70)
        
        # Basic statistics
        print(f"\nBasic Statistics:")
        print(f"  Trajectory length:    {len(self.phase_history)} steps")
        print(f"  Mean phase diff:      {np.mean(self.phase_history):.4f} rad ({np.mean(self.phase_history)*180/math.pi:.2f}°)")
        print(f"  Phase diff std dev:   {np.std(self.phase_history):.4f} rad")
        print(f"  Phase diff range:     [{np.min(self.phase_history):.4f}, {np.max(self.phase_history):.4f}]")
        
        # Parameter statistics
        print(f"\nLearned Parameters:")
        print(f"  Final phase_velocity:  {self.velocity_history[-1]:.4f}")
        print(f"  Final phase_amplitude: {self.amplitude_history[-1]:.4f}")
        print(f"  Velocity range:        [{np.min(self.velocity_history):.4f}, {np.max(self.velocity_history):.4f}]")
        print(f"  Amplitude range:       [{np.min(self.amplitude_history):.4f}, {np.max(self.amplitude_history):.4f}]")
        
        # Exploration statistics
        print(f"\nExploration Performance:")
        print(f"  Mean diversity:       {np.mean(self.diversity_history):.4f}")
        print(f"  Mean reward:          {np.mean(self.reward_history):.2f}")
        print(f"  Unique state pairs:   {len(set(zip(self.cw_history, self.ccw_history)))}")
        
        # Phase attractor detection
        phase_bins = np.histogram(self.phase_history, bins=8, range=(0, 2*math.pi))[0]
        attractors = [i for i, count in enumerate(phase_bins) if count > len(self.phase_history) * 0.15]
        
        print(f"\nPhase Attractors:")
        if attractors:
            for idx in attractors:
                phase_center = (idx + 0.5) * 2 * math.pi / 8
                percentage = phase_bins[idx] / len(self.phase_history) * 100
                print(f"  Attractor {idx+1}: {phase_center:.4f} rad ({phase_center*180/math.pi:.1f}°) - {percentage:.1f}%")
        else:
            print("  No significant attractors detected (uniform phase distribution)")
        
        print("\n" + "="*70 + "\n")

def main():
    print("="*70)
    print("  Phase Bifurcation Visualization & Analysis Tool")
    print("="*70)
    
    # Load or create model
    model = BifurcationNetwork()
    
    try:
        model.load_state_dict(torch.load('bifurcation_enhanced_v2.pt'))
        print("\n✓ Loaded trained model")
    except FileNotFoundError:
        print("\n⚠ Model not found, using random initialization")
        print("  Recommend running bifurcation_enhanced_v2.py first")
    
    # Create environment and analyzer
    env = BifurcationEnv(max_steps=1000)
    analyzer = PhaseAnalyzer(model, env)
    
    # Collect data
    print("\nCollecting trajectory data...")
    analyzer.collect_trajectory(num_steps=1000, temperature=1.5, noise_scale=0.2)
    
    # Generate all visualizations
    print("\nGenerating visualization plots...")
    analyzer.plot_phase_bifurcation()
    analyzer.plot_ripple_dynamics()
    analyzer.plot_parameter_impact()
    analyzer.plot_phase_portrait()
    
    # Generate report
    analyzer.generate_summary_report()
    
    print("✓ All analyses complete!")
    print("\nGenerated files:")
    print("  - phase_bifurcation.png  (Phase bifurcation diagram)")
    print("  - ripple_dynamics.png    (Ripple dynamics)")
    print("  - parameter_impact.png   (Parameter impact)")
    print("  - phase_portrait.png     (Phase portrait)")

if __name__ == "__main__":
    main()
