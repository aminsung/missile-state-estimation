#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Min Sung Ahn"
__date__ = "13 MAR 2016"

# ============================================================================
# Global Constants
# ============================================================================
V_C = 300.0 # [ft/sec]
T_F = 10.0 # [sec]
R1 = 15.0e-6 # [rad^2 / sec]
R2 = 1.67e-3 # [rad^2 / sec^3]
TAU = 2.0 # [sec]
W = 100.0e2 # [(ft/sec^2)^2]
DT = 0.01 # [sec]
T = np.arange(0.0, 10.0, DT) # [sec]

# Dynamic State
F = np.array([[0.0, 1.0, 0.0], [0.0, 0.0 -1.0], [0.0, 0.0, -(1/TAU)]])
G = np.array([[0], [0], [1]])

# ============================================================================
# Kalman Filter
# Status: WIP
# ============================================================================
P_init = np.array([[0.0, 0.0, 0.0], [0.0, 200.0e2, 0.0], [0.0, 0.0, 100.0e2]])
P_history = np.zeros((3, 3, T.size))
K_history = np.zeros((3, 3, T.size))
Z_history = np.zeros(T.size)
x_hat_history = np.zeros((3, 1, T.size))
dx_hat_history = np.zeros((3, 1, T.size))
x_history = np.zeros((3, 1, T.size))
dx_history = np.zeros((3, 1, T.size))
err_history = np.zeros((3, 1, T.size))
residual = np.zeros((2, 1))


print("Missile State Estimation Done")