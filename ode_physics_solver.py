"""
Scientific / Math Calculator
----------------------------
A small command-line tool that solves differential equations and simple
physics problems numerically, then plots the results.

Modes:
  1. Damped harmonic oscillator      x'' + 2*zeta*w0*x' + w0^2*x = 0
  2. Projectile motion with air drag
  3. Custom first-order ODE          dy/dt = f(t, y)
  4. Custom second-order ODE         y''  = f(t, y, v)   (v = y')

Requirements: numpy, scipy, matplotlib, sympy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp


# ----------------------------------------------------------------- helpers
def ask_float(prompt, default):
    """Ask the user for a number; fall back to the default on empty/invalid input."""
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"  Invalid number, using default ({default}).")
        return default


def parse_expression(text, symbols):
    """Turn a string like '-2*y + sin(t)' into a fast numpy function."""
    t, y, v = sp.symbols("t y v")
    local = {"t": t, "y": y, "v": v}
    expr = sp.sympify(text, locals=local)
    return sp.lambdify(symbols(t, y, v), expr, modules="numpy")


def finish_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"  Plot saved to {filename}")
    plt.show()


# ------------------------------------------------------------------ modes
def damped_oscillator():
    print("\nDamped harmonic oscillator:  x'' + 2*zeta*w0*x' + w0^2*x = 0")
    w0 = ask_float("Natural frequency w0 (rad/s)", 2.0)
    zeta = ask_float("Damping ratio zeta", 0.1)
    x0 = ask_float("Initial position x0", 1.0)
    v0 = ask_float("Initial velocity v0", 0.0)
    t_end = ask_float("End time (s)", 20.0)

    def rhs(t, s):
        x, v = s
        return [v, -2 * zeta * w0 * v - w0**2 * x]

    sol = solve_ivp(rhs, (0, t_end), [x0, v0],
                    t_eval=np.linspace(0, t_end, 1000), rtol=1e-9, atol=1e-11)

    if zeta < 1:
        regime = "underdamped"
    elif zeta == 1:
        regime = "critically damped"
    else:
        regime = "overdamped"
    print(f"  Regime: {regime}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(sol.t, sol.y[0])
    ax1.set(title=f"Position vs time ({regime})", xlabel="t (s)", ylabel="x(t)")
    ax1.grid(True)
    ax2.plot(sol.y[0], sol.y[1])
    ax2.set(title="Phase portrait", xlabel="x", ylabel="v")
    ax2.grid(True)
    finish_plot(fig, "damped_oscillator.png")


def projectile():
    print("\nProjectile motion with quadratic air drag")
    v0 = ask_float("Launch speed (m/s)", 50.0)
    angle = ask_float("Launch angle (degrees)", 45.0)
    h0 = ask_float("Initial height (m)", 0.0)
    k = ask_float("Drag coefficient k/m (1/m), 0 = no drag", 0.01)
    g = 9.81

    th = np.radians(angle)

    def rhs(t, s):
        x, y, vx, vy = s
        speed = np.hypot(vx, vy)
        return [vx, vy, -k * speed * vx, -g - k * speed * vy]

    def hit_ground(t, s):
        return s[1]

    hit_ground.terminal = True
    hit_ground.direction = -1

    sol = solve_ivp(rhs, (0, 1000), [0, h0, v0 * np.cos(th), v0 * np.sin(th)],
                    events=hit_ground, dense_output=True, rtol=1e-9, atol=1e-11)

    t_flight = sol.t[-1]
    ts = np.linspace(0, t_flight, 500)
    x, y = sol.sol(ts)[0], sol.sol(ts)[1]
    print(f"  Flight time : {t_flight:.3f} s")
    print(f"  Range       : {x[-1]:.3f} m")
    print(f"  Max height  : {y.max():.3f} m")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y)
    ax.set(title="Projectile trajectory", xlabel="x (m)", ylabel="y (m)")
    ax.set_ylim(bottom=0)
    ax.grid(True)
    finish_plot(fig, "projectile.png")


def custom_first_order():
    print("\nCustom first-order ODE:  dy/dt = f(t, y)")
    print("  Example: -2*y + sin(t)")
    text = input("f(t, y) = ").strip() or "-2*y + sin(t)"
    y0 = ask_float("Initial value y(t0)", 1.0)
    t0 = ask_float("Start time t0", 0.0)
    t_end = ask_float("End time", 10.0)

    try:
        f = parse_expression(text, lambda t, y, v: (t, y))
    except Exception as e:
        print(f"  Could not parse expression: {e}")
        return

    sol = solve_ivp(lambda t, y: [f(t, y[0])], (t0, t_end), [y0],
                    t_eval=np.linspace(t0, t_end, 800), rtol=1e-9, atol=1e-11)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sol.t, sol.y[0])
    ax.set(title=f"dy/dt = {text}", xlabel="t", ylabel="y(t)")
    ax.grid(True)
    finish_plot(fig, "custom_first_order.png")


def custom_second_order():
    print("\nCustom second-order ODE:  y'' = f(t, y, v)   where v = y'")
    print("  Example (Van der Pol): 1.5*(1 - y**2)*v - y")
    text = input("f(t, y, v) = ").strip() or "1.5*(1 - y**2)*v - y"
    y0 = ask_float("Initial y", 2.0)
    v0 = ask_float("Initial v = y'", 0.0)
    t_end = ask_float("End time", 30.0)

    try:
        f = parse_expression(text, lambda t, y, v: (t, y, v))
    except Exception as e:
        print(f"  Could not parse expression: {e}")
        return

    sol = solve_ivp(lambda t, s: [s[1], f(t, s[0], s[1])], (0, t_end), [y0, v0],
                    t_eval=np.linspace(0, t_end, 2000), rtol=1e-9, atol=1e-11)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(sol.t, sol.y[0])
    ax1.set(title=f"y'' = {text}", xlabel="t", ylabel="y(t)")
    ax1.grid(True)
    ax2.plot(sol.y[0], sol.y[1])
    ax2.set(title="Phase portrait", xlabel="y", ylabel="y'")
    ax2.grid(True)
    finish_plot(fig, "custom_second_order.png")


# ------------------------------------------------------------------- main
MENU = {
    "1": ("Damped harmonic oscillator", damped_oscillator),
    "2": ("Projectile motion with air drag", projectile),
    "3": ("Custom first-order ODE", custom_first_order),
    "4": ("Custom second-order ODE", custom_second_order),
}


def main():
    print("=== Scientific / Math Calculator ===")
    while True:
        print()
        for key, (name, _) in MENU.items():
            print(f"  {key}. {name}")
        print("  q. Quit")
        choice = input("Choose: ").strip().lower()
        if choice == "q":
            print("Goodbye!")
            break
        if choice in MENU:
            MENU[choice][1]()
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
