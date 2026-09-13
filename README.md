# Stochastic Differential Equations Approaches to Quantum System Noise Analysis and Control

This repository contains the manuscript, simulation code, and figures for the paper *"Stochastic Differential Equations Approaches to Quantum System Noise Analysis and Control."*

**Authors:** Kumar Gautam, Akshit Dutta, Nikhil Pachauri, Namisha Gupta, Kaumud Sharma

## Overview

The paper develops a mathematical and computational framework for analyzing classical noise in quantum systems using Itô stochastic calculus. The noise-modulating potential is modeled as a linear combination of known Hermitian operators with unknown real coefficients, and the unitary time-evolution operator is expanded to second order in the perturbation strength (O(ε²)) using Itô stochastic integrals with respect to Brownian motion.

A key result is a closed-form quadratic functional Q(θ) — derived from the eigenprojectors of the system Hamiltonian — that links time-averaged observable measurements to the unknown noise parameters θ. The paper shows this inverse problem is degenerate for a single observable (the solution set is a one-dimensional curve in parameter space) and demonstrates, both analytically and numerically, how a second, linearly independent observable breaks the degeneracy and substantially improves parameter recovery.

## Key Contributions

- Operator-based vector potential model: noise potential represented as a linear function of unknown parameters with Hermitian operator coefficients
- Second-order (O(ε²)) perturbative expansion of the unitary evolution operator via first- and second-order (doubly iterated) Itô stochastic integrals
- Derivation of a closed-form quadratic functional Q(θ) from the eigenprojector decomposition of the Hamiltonian
- Explicit treatment of the single-observable degeneracy in the noise-parameter inverse problem, and its resolution using multiple observables
- A full numerical validation: Euler–Maruyama SDE simulation of a single-qubit system, Monte Carlo estimation of Q(θ), and quantitative comparison against the analytical prediction
- Relation of the framework to established quantum stochastic/quantum-optics literature (Gardiner–Zoller, Carmichael) and to physical realizations in QKD, quantum repeater networks, and quantum teleportation

## Repository Contents

```
.
├── paper/              # Manuscript source (LaTeX) and compiled PDF
├── simulations/        # Python simulation code (Euler–Maruyama SDE integration, Q(θ) estimation)
├── figures/            # Generated figures (Brownian motion paths, observable evolution, Q(θ) landscape, etc.)
└── README.md
```

*(Adjust the folder names above to match the actual repository layout.)*

## Simulation Details

The numerical results were produced in **Python 3.12** using:

- **NumPy** — matrix algebra
- **SciPy** — exact matrix-exponential computation
- **Matplotlib** — visualization

Simulation setup (single-qubit system):

| Quantity | Value |
|---|---|
| Hamiltonian, H | ½σ_z + (3/10)σ_x |
| Noise operators | V₁ = σ_z/√2, V₂ = σ_y/√2 |
| True parameters, θ | (0.40, 0.60) |
| Perturbation strength, ε | 0.15 |
| Time span / steps | [0, 8], N = 4000 (Δt = 0.002) |
| Monte Carlo paths, N_MC | 8000 |
| Observable | σ_z (and σ_x for the two-observable case) |

All stochastic realizations use a fixed random seed for full reproducibility.

### Reproducing the Results

```bash
# example — update to match actual script names
pip install numpy scipy matplotlib
python simulations/run_sde_simulation.py
```

This generates the Brownian motion sample paths, the noise-free and noise-perturbed observable evolution, the rescaled correction term c(t), the analytical Q(θ₁, θ₂) landscape, and the single- and two-observable parameter estimates reported in the paper.

## Citation

If you use this work, please cite:

```bibtex
@article{gautam_sde_quantum_noise,
  title   = {Stochastic Differential Equations Approaches to Quantum System Noise Analysis and Control},
  author  = {Gautam, Kumar and Dutta, Akshit and Pachauri, Nikhil and Gupta, Namisha and Sharma, Kaumud},
  note    = {Revised manuscript},
}
```

*(Update with the final venue, volume, page numbers, and DOI once published.)*

## Acknowledgements

The authors thank Prof. K. R. Parthasarathy for his guidance and mentorship throughout the development of this work.

## Funding

Manipal Institute of Technology, Manipal Academy of Higher Education
