# Physics-Informed Neural Networks

A hands-on exploration of **Physics-Informed Neural Networks (PINNs)** using PyTorch, developed while learning Physics-Informed Machine Learning concepts during my research internship experience at IIT Bombay.

The repository documents my progression from solving simple differential equations to implementing a PINN for the **1D Heat Equation**.

## 📌 Learning Progression

### 01 — Simple Differential Equation

Started with a simple ordinary differential equation to understand how a neural network can learn a solution while being constrained by the governing equation.

Key concepts:

* Neural network approximation
* Automatic differentiation
* Physics loss
* Boundary conditions
* Adam optimization

### 02 — Partial Differential Equation

Extended the approach from an ODE to a PDE with spatial and temporal variables.

Implemented:

* Space-time collocation points
* Automatic differentiation
* PDE residual calculation
* Physics-based loss
* Neural network optimization

### 03 — 1D Heat Equation

Implemented a complete PINN for the one-dimensional heat equation:

$$
\frac{\partial u}{\partial t}
=
\alpha
\frac{\partial^2 u}{\partial x^2}
$$

The network learns the temperature distribution while satisfying the governing physics, initial condition, and boundary conditions.

The notebook includes:

* Physics collocation points
* Initial conditions
* Boundary conditions
* First- and second-order derivatives using PyTorch Autograd
* Physics loss
* Initial-condition loss
* Boundary-condition loss
* Total loss
* Adam optimization
* Analytical vs PINN solution comparison
* Error heatmap
* Time-slice comparisons
* 3D visualization of the learned solution

## 🧠 Concepts Explored

* Physics-Informed Neural Networks
* Physics-Informed Machine Learning
* Automatic Differentiation
* PDE Residuals
* Collocation Points
* Initial & Boundary Conditions
* Physics-Based Loss Functions
* PDE-Constrained Optimization
* PyTorch

## 🛠️ Tech Stack

* Python
* PyTorch
* NumPy
* Matplotlib
* Jupyter Notebook

## 📊 Final Experiment

The final notebook compares the PINN prediction with the analytical solution of the 1D Heat Equation and visualizes the resulting prediction error across space and time.

This progression helped me understand how neural networks can incorporate mathematical and physical constraints directly into the learning process instead of relying entirely on labeled training data.

## 📁 Repository Structure

```text
notebooks/
├── 01_Simple_DE_PINN.ipynb
├── 02_PDE_PINN.ipynb
└── 03_1D_Heat_Equation_PINN.ipynb
```

> This repository represents my learning and experimental work with PINNs and is intended as a technical record of the concepts and implementations I explored.
