# Emergent Phase Dynamics in Reinforcement Learning
> A Python framework exploring multi-attractor phase dynamics in counter-rotating oscillators under policy gradient optimization.

## Overview
This repository demonstrates **emergent, unpinned phase dynamics** in a dual CW/CCW discrete-continuous RL environment. The system uses learnable phase velocity and amplitude parameters to explore quasi-periodic and itinerant phase behavior without enforcing explicit phase locking.

The framework consists of four main modules:
1. **advanced_double_helix_tuner.py** – Core RL agent trainer with dual oscillators.
2. **bifurcation_enhanced_v2.py** – Enhanced bifurcation environment with reward shaping, entropy control, and phase variance monitoring.
3. **phase_bifurcation_analyzer.py** – Analytical tools to evaluate phase differences, variance, and emergent attractors.
4. **live_phase_demo.py** – Real-time visualization of phase evolution, including static and dynamic charts, polar plots, and reward monitoring.

---

## Features
- Dual CW/CCW agent controlling 64×64 discrete state space
- Learnable phase parameters: velocity & amplitude
- Entropy-driven exploration to prevent mode collapse
- Emergent itinerant phase dynamics without hard-coded targets
- Real-time visualizations and static analytics
- Cross-platform CPU and GPU support (tested on PyTorch, OpenCL-friendly)

---

## Installation
```bash
git clone https://github.com/yourusername/emergent-phase-rl.git
cd emergent-phase-rl
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt


Usage
Training
python bifurcation_enhanced_v2.py

Real-time Phase Demo
python live_phase_demo.py --frames 300 --save phase_animation.gif

Analysis
python phase_bifurcation_analyzer.py

Results

Average phase difference: ~1.6–2.0 rad, no π-locking

High state diversity (~70–99% unique CW/CCW pairs)

Emergent multi-attractor dynamics

GPU-friendly (tested on CPU first; AMD ROCm & NVIDIA possible)

Citation

If you use this repository, please cite as:

Yufan (2026). Emergent Multi-Attractor Phase Dynamics in Reinforcement Learning-Controlled Counter-Rotating Os
