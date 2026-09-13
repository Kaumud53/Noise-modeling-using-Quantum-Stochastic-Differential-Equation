# Stochastic Differential Equations Approaches to Quantum System Noise Analysis and Control

## Overview
This repository contains the theoretical framework, algorithms, and simulation code associated with the paper **"Stochastic Differential Equations Approaches to Quantum System Noise Analysis and Control"** by Kumar Gautam, Akshit Dutta, Nikhil Pachauri, Namisha Gupta, and Kaumud Sharma. 

The project presents a detailed analysis of an Itô-type stochastic differential equation in vector space to model standard noise-induced effects in quantum systems. By representing the vector potential as a linear function of unknown parameters with Hermitian operators as coefficients, this framework provides a refined mathematical approach for analyzing and mitigating classical noise effects in quantum computing and control.

## Key Contributions
- **Operator-Based Formulation:** Systematically models classical noise using a vector potential represented as a linear function of unknown parameters with Hermitian operators.
- **Higher-Order Perturbation:** Analyzes the unitary time-evolution operator using a second-order perturbation parameter, expressing the $\mathcal{O}(\epsilon)$ and $\mathcal{O}(\epsilon^2)$ terms as Itô stochastic integrals and doubly iterated stochastic integrals with respect to Brownian motion, respectively.
- **Parameter Estimation:** Implements an indirect, multi-parameter estimation framework using time-averaged observables and quadratic forms to accurately characterize noise without the overhead of full quantum tomography.
- **Noise Control:** Delivers a robust mathematical basis to improve the stability and performance of quantum devices by addressing environmental disturbances and supporting physical realizations in systems like Quantum Key Distribution (QKD) and distributed quantum networks.

## Mathematical Framework
The core evolution of the quantum system under white Gaussian noise is governed by the Itô stochastic differential equation:
```math
dU(t) = \left(-iH dt - \frac{\epsilon^2}{2}V^2 dt - i\epsilon V dB(t)\right)U(t)
```
Where:
- `H` is the free Hamiltonian.
- `V` is the noise modulating potential (linear combination of known operators with unknown parameters $\theta_k$).
- `B(t)` represents standard Brownian motion.
- `\epsilon` is the perturbation parameter.

## Algorithm
The repository implements **Algorithm 1: Method for Estimating Quantum Parameters for Noise Control**. This algorithm iterates through parameter estimation using Monte Carlo simulations of Brownian motion paths, calculating first and second-order stochastic integrals to update the unitary evolution operator. A grid-search optimization is then applied to the resulting quadratic function $Q(\theta)$ to infer the unknown noise parameters.

## Dependencies
The numerical simulations and Monte Carlo integrations are built on the following stack:
- Python 3.12
- `numpy` (for matrix algebra and path generation)
- `scipy` (for exact matrix-exponential computation)
- `matplotlib` (for landscape and trajectory visualization)

## Usage
*(Placeholder for repository structure and run instructions)*
```bash
# Clone the repository
git clone https://github.com/yourusername/quantum-sde-noise.git
cd quantum-sde-noise

# Install dependencies
pip install numpy scipy matplotlib

# Run the parameter estimation simulation
python simulate.py
```

## Authors
- Kumar Gautam
- Akshit Dutta
- Nikhil Pachauri
- Namisha Gupta
- Kaumud Sharma

## Acknowledgements
We acknowledge the foundational contributions to quantum stochastic calculus and the theoretical frameworks that enable advanced quantum parameter estimation.
