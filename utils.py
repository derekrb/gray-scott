### NE 451 Final Project - Gray-Scott Reaction/Diffusion ###

# Submitted by: Derek Bennewies
# Last Edited: November 29, 2012

import numpy as np

# Utility Functions #


# Compute the 5-point Laplacian with PBC - O(h^2)
def lap5(X, h2):

    # Compute the stencil matrices
    X_up = np.roll(X, 1, axis=1)
    X_down = np.roll(X, -1, axis=1)
    X_left = np.roll(X, 1, axis=0)
    X_right = np.roll(X, -1, axis=0)

    # Compute the Laplacian
    lap = (X_up + X_down + X_left + X_right - 4 * X) / h2

    return lap


# Compute the 9-point Laplacian with PBC - O(h^2)
def lap9(X, h2):

    # Compute adjacent the stencil matrices
    X_up = np.roll(X, 1, axis=1)
    X_down = np.roll(X, -1, axis=1)
    X_left = np.roll(X, 1, axis=0)
    X_right = np.roll(X, -1, axis=0)

    # Compute the diagonal stencil matrices
    X_r_up = np.roll(X_right, 1, axis=1)
    X_r_down = np.roll(X_right, -1, axis=1)
    X_l_up = np.roll(X_left, 1, axis=1)
    X_l_down = np.roll(X_left, -1, axis=1)

    # Compute the Laplacian
    lap = ((X_up + X_down + X_left + X_right) * 2 / 3 + \
        (X_l_up + X_l_down + X_r_up + X_r_down) * 1 / 6 - \
        X * 10 / 3) / h2

    return lap


# Evaluate the concentration of u given parameters
def u_fun(grid_u, lap_u, du, grid_uvv, feed):
    return du * lap_u - grid_uvv + feed * (1 - grid_u)


# Evaluate the concentration of v given parameters
def v_fun(grid_v, lap_v, grid_uvv, dv, feed, k):
    return dv * lap_v + grid_uvv - (feed + k) * grid_v


# Explicit Euler integration - O(h)
def exp_euler(y, dt, fun):
    return dt * fun(y)


# Midpoint Method integration - O(h^2)
def midpoint(y, dt, fun):

    k1 = dt * 0.5 * fun(y)

    return dt * fun(y + k1)


# Runge-Kutta integration - O(h^4)
def rk4(y, dt, fun):

    k1 = dt * fun(y)
    k2 = dt * fun(y + 0.5 * k1)
    k3 = dt * fun(y + 0.5 * k2)
    k4 = dt * fun(y + k3)

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
