# ODE Solver

A command-line tool that numerically solves ordinary differential equations (first and second order) and plots the solution.

## Features
- First-order ODE: `dy/dt = f(t, y)`
- Second-order ODE: `y'' = f(t, y, v)` where `v = y'`

## Installation
```bash
pip install numpy scipy matplotlib sympy
```

## Usage
```bash
python ode_physics_solver.py
```
Choose a mode, type your equation, enter the initial conditions (press Enter for defaults), and the plot is shown and saved as a PNG.

## Examples
- First-order: `-2*y + sin(t)`
- Second-order (damped spring): `-4*y - 0.5*v`

## Method
Equations are solved with SciPy's `solve_ivp`. A second-order equation is converted into a system of two first-order equations: `y' = v`, `v' = f(t, y, v)`.
