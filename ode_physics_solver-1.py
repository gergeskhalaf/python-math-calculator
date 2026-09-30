"""
ODE Solver
----------
A simple command-line tool that solves differential equations numerically
and plots the solution.

Modes:
  1. First-order ODE    dy/dt = f(t, y)
  2. Second-order ODE   y''   = f(t, y, v)   (where v = y')

Requirements: numpy, scipy, matplotlib, sympy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp


def ask_float(prompt, default):
    """Ask for a number; use the default if the input is empty or invalid."""
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"  Invalid number, using default ({default}).")
        return default


def parse_expression(text, variables):
    """Convert a string like '-2*y + sin(t)' into a numpy function."""
    t, y, v = sp.symbols("t y v")
    expr = sp.sympify(text, locals={"t": t, "y": y, "v": v})
    symbols = {"t": t, "y": y, "v": v}
    return sp.lambdify([symbols[name] for name in variables], expr, modules="numpy")


def first_order():
    print("\nFirst-order ODE:  dy/dt = f(t, y)")
    print("  Example: -2*y + sin(t)")
    text = input("f(t, y) = ").strip() or "-2*y + sin(t)"
    y0 = ask_float("Initial value y(0)", 1.0)
    t_end = ask_float("End time", 10.0)

    try:
        f = parse_expression(text, ["t", "y"])
    except Exception as e:
        print(f"  Could not read the equation: {e}")
        return

    sol = solve_ivp(lambda t, y: [f(t, y[0])], (0, t_end), [y0],
                    t_eval=np.linspace(0, t_end, 800))

    plt.figure(figsize=(8, 4.5))
    plt.plot(sol.t, sol.y[0])
    plt.title(f"dy/dt = {text}")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.grid(True)
    plt.savefig("first_order.png", dpi=150)
    print("  Plot saved to first_order.png")
    plt.show()


def second_order():
    print("\nSecond-order ODE:  y'' = f(t, y, v)   where v = y'")
    print("  Example (spring): -4*y - 0.5*v")
    text = input("f(t, y, v) = ").strip() or "-4*y - 0.5*v"
    y0 = ask_float("Initial value y(0)", 1.0)
    v0 = ask_float("Initial velocity y'(0)", 0.0)
    t_end = ask_float("End time", 20.0)

    try:
        f = parse_expression(text, ["t", "y", "v"])
    except Exception as e:
        print(f"  Could not read the equation: {e}")
        return

    # Turn y'' = f into a system: y' = v, v' = f(t, y, v)
    sol = solve_ivp(lambda t, s: [s[1], f(t, s[0], s[1])], (0, t_end), [y0, v0],
                    t_eval=np.linspace(0, t_end, 1000))

    plt.figure(figsize=(8, 4.5))
    plt.plot(sol.t, sol.y[0])
    plt.title(f"y'' = {text}")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.grid(True)
    plt.savefig("second_order.png", dpi=150)
    print("  Plot saved to second_order.png")
    plt.show()


def main():
    print("=== ODE Solver ===")
    while True:
        print("\n  1. First-order ODE")
        print("  2. Second-order ODE")
        print("  q. Quit")
        choice = input("Choose: ").strip().lower()
        if choice == "1":
            first_order()
        elif choice == "2":
            second_order()
        elif choice == "q":
            print("Goodbye!")
            break
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
