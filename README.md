# Scientific / Math Calculator

A command-line tool that numerically solves differential equations and simple physics problems, then plots the results.

## Features
- Damped harmonic oscillator (time series + phase portrait)
- Projectile motion with quadratic air drag (range, max height, flight time)
- Custom first-order ODE: `dy/dt = f(t, y)`
- Custom second-order ODE: `y'' = f(t, y, v)` (e.g. Van der Pol)

## Installation
```bash
pip install numpy scipy matplotlib sympy
```

## Usage
```bash
python ode_physics_solver.py
```
Pick a mode from the menu, enter the parameters (press Enter to accept defaults), and the plot is shown and saved as a PNG.

## Example input
- First-order: `-2*y + sin(t)`
- Second-order: `1.5*(1 - y**2)*v - y`

## Method
Equations are integrated with SciPy's `solve_ivp` (Runge-Kutta 4(5), adaptive step size).
