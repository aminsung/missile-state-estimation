#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Min Sung Ahn"
__date__ = "13 MAR 2016"


# ============================================================================
# Global Constants
# ============================================================================
V_C = 300.0  # [ft/sec]
T_F = 10.0  # [sec]
R1 = 15.0*(10.0**(-6))  # [rad^2 sec]
R2 = 1.67*(10.0**(-3))  # [rad^2 sec^3]
TAU = 2.0  # [sec]
W = 100.0**2  # [(ft/sec^2)^2]
DT = 0.01  # [sec]
T = np.arange(0.0, 10.0, DT)  # [sec]
A_MEAN = 0.0
A_VAR = 100.0**2
Y_MEAN = 0.0
Y_VAR = 0.0**2
V_MEAN = 0.0
V_VAR = 200.0**2


# Ref. Stochastic Processes, Estimation, and Control pg. 313
# by Jason L. Speyer & Walter H. Chung
# ============================================================================
# State Space
# ============================================================================
F = np.array([[0.0, 1.0, 0.0],
              [0.0, 0.0, -1.0],
              [0.0, 0.0, -(1/TAU)]])
B = np.array([[0], [1], [0]])
G = np.array([[0], [0], [1]])


# ============================================================================
# Initial Values
# ============================================================================
H_init = np.array([[1.0/(V_C*T_F), 0.0, 0.0]])
V_init = R1 + (R2/(T_F**2))
y_init = 0.0
v_init = np.random.normal(V_MEAN, np.sqrt(V_VAR))
at_init = np.random.normal(0.0, np.sqrt(A_VAR))

N_MEAN = 0.0
N_VAR = V_init/DT
n_init = np.random.normal(N_MEAN, np.sqrt(N_VAR))

W_at = np.random.normal(A_MEAN, np.sqrt(A_VAR/DT))


# ============================================================================
# Kalman Filter
# Status: WIP
# ============================================================================
P_init = np.array([[0.0, 0.0, 0.0],
                   [0.0, 200.0**2, 0.0],
                   [0.0, 0.0, 100.0**2]])
P_history = np.zeros((3, 3, T.size))
K_history = np.zeros((3, 1, T.size))
Z_history = np.zeros(T.size)
x_hat_history = np.zeros((3, 1, T.size))
dx_hat_history = np.zeros((3, 1, T.size))
x_history = np.zeros((3, 1, T.size))
dx_history = np.zeros((3, 1, T.size))
err_history = np.zeros((3, 1, T.size))
residual = np.zeros((2, 1))

P_history[:, :, 0] = P_init
K_history[:, :, 0] = np.dot(P_init, H_init.transpose())*(V_init**(-1))

# Initialize ACTUAL states
x_history[:, :, 0] = np.array([[y_init], [v_init], [at_init]])
dx_history[:, :, 0] = np.dot(F, x_history[:, :, 0]) + np.dot(G, W_at)
Z[0] = np.dot(H_init, x_history[:, :, 0]) + n_init  # z = theta + n

# Estimate states
x_hat_history[:, :, 0] = np.array([[0, 0, 0]]).transpose()
dx_hat_history[:, :, 0] = np.dot(F, x_hat_history[:, :, 1]) + \
                          K_history[:, :, 0]*(Z_history[0] - \
                                              np.dot(H_init, x_hat_history[:, :, 1]))
err_history[:, :, 0] = np.array([[0, 0, 0]]).transpose()








print("Missile State Estimation Done")