# Emergent Phase Dynamics in Reinforcement Learning

A research‐oriented Python framework for exploring **emergent phase dynamics** in a dual CW/CCW oscillator
environment under Reinforcement Learning, exhibiting multi‐attractor and itinerant behavior without explicit phase pinning.

This repository is designed for **reproducible experimentation** and includes full visualization and analysis tooling.

---

## 🔍 Project Overview

Traditional RL systems often converge prematurely or exhibit mode collapse.  
Here we investigate **phase‐based dynamics**, where:

- Phase parameters are **learnable**
- Phase differences **do not lock** to fixed targets
- Phase evolution exhibits **quasi‐periodic and multi‐attractor behavior**
- No forced or hand‐coded phase constraints

This framework works on **CPUs and GPUs (NVIDIA/AMD)** and does **not rely on closed, vendor‐locked ecosystems**.

---

## 🧩 Repository Structure

| File | Description |
|------|-------------|
| `advanced_double_helix_tuner.py` | Core phase training module (episodic RL with phase coupling) |
| `bifurcation_enhanced_v2.py` | Enhanced model generation with bifurcation control |
| `phase_bifurcation_analyzer.py` | Static & dynamic analysis + report generation |
| `live_phase_demo.py` | Animation generator to visualize phase evolution |

---

## 🚀 Quick Start

Install dependencies:

python3 -m venv .venv
.venv\Scripts\activate         # Windows

pip install -r requirements.txt
Run Training

python bifurcation_enhanced_v2.py
📊 Visualization & Analysis
Generate the full analysis report and figures:

python phase_bifurcation_analyzer.py
Expected output:

Phase bifurcation diagram saved to phase_bifurcation.png
Ripple dynamics plot saved to ripple_dynamics.png
Parameter impact analysis saved to parameter_impact.png
Phase portrait saved to phase_portrait.png
...
This will produce:

phase_bifurcation.png (phase time series)

ripple_dynamics.png (ripple & FFT)

parameter_impact.png (parameter influence)

phase_portrait.png (phase portraits, return map, chaos indicators)

🧪 Live Demo
Display real‐time evolution (optional):

python live_phase_demo.py
Save animations:

python live_phase_demo.py --frames 300 --save phase_animation.gif
📈 Results Summary (Example)
yaml

Basic Statistics:
  Trajectory length: 1000 steps
  Mean phase diff:     1.8813 rad
  Phase diff std dev:  0.4454 rad

Learned Parameters:
  phase_velocity: 0.4951
  phase_amplitude: 0.9954

Exploration Performance:
  Mean diversity: 0.8824
  Unique pairs:   807
🧠 Key Concepts
Emergent Phase Dynamics
Phase differences evolve without explicit pinning or loss functions.

Multi‑Attractor Behavior
The system exhibits multiple attractors and controlled bifurcation exploration.

Hardware Agnostic
Works on CPU & GPU, without fixing to a specific vendor stack.

📌 License
MIT © 2026 RetryixAGI
