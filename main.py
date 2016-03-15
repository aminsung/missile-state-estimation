#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Min Sung Ahn"
__date__ = "13 MAR 2016"


# ============================================================================
# Global Configurations
# ============================================================================
display_plot = False
display_plot = True
single_run_plot = False
single_run_plot = True


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
              [0.0, 0.0, -(1.0/TAU)]])
B = np.array([[0.0], [1.0], [0.0]])
G = np.array([[0.0], [0.0], [1.0]])


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
n = np.random.normal(N_MEAN, np.sqrt(N_VAR))  # Noise

W_at = np.random.normal(A_MEAN, np.sqrt(A_VAR/DT))


# ============================================================================
# Kalman Filter
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
residual = np.zeros((1, T.size))

P_history[:, :, 0] = P_init
K_history[:, :, 0] = np.dot(np.dot(P_init, H_init.transpose()), (V_init**(-1)))

# Initialize actual states
x_history[:, :, 0] = np.array([[y_init], [v_init], [at_init]])
dx_history[:, :, 0] = np.dot(F, x_history[:, :, 0]) + np.dot(G, W_at)
Z_history[0] = np.dot(H_init, x_history[:, :, 0]) + n  # z = theta + n

# Initialize estimate states
x_hat_history[:, :, 0] = np.array([[0, 0, 0]]).transpose()
dx_hat_history[:, :, 0] = np.dot(F, x_hat_history[:, :, 0]) + \
                          np.dot(K_history[:, :, 0], \
                          (Z_history[0] - np.dot(H_init, x_hat_history[:, :, 0])))
err_history[:, :, 0] = np.array([[0, 0, 0]]).transpose()

for idx in range(0, len(T)-1):
    H = np.array([[1.0/(V_C*(T_F-T[idx])), 0.0, 0.0]])
    V = R1 + (R2/((T_F-T[idx])**2))

    # Update Variance
    P_dot = np.dot(F, P_history[:, :, idx]) + \
            np.dot(P_history[:, :, idx], F.transpose()) - \
            np.dot(np.dot(np.dot(np.dot(P_history[:, :, idx], H.transpose()), V**(-1)), H), P_history[:, :, idx]) + \
            np.dot(np.dot(G, W), G.transpose())
    P_history[:, :, idx+1] = P_history[:, :, idx] + np.dot(P_dot, DT)

    # Update Kalman Gain
    K_history[:, :, idx+1] = np.dot(np.dot(P_history[:, :, idx], H.transpose()), (V**(-1)))

    # Update with new iteration of noise
    n = np.random.normal(N_MEAN, np.sqrt(V/DT))
    W_at = np.random.normal(A_MEAN, np.sqrt(A_VAR/DT))

    # Update actual states
    dx_history[:, :, idx+1] = np.dot(F, x_history[:, :, idx]) + \
                              np.dot(G, W_at)
    x_history[:, :, idx+1] = x_history[:, :, idx] + dx_history[:, :, idx+1]*DT
    Z_history[idx+1] = np.dot(H, x_history[:, :, idx+1]) + n

    # Update estimate states
    dx_hat_history[:, :, idx+1] = np.dot(F, x_hat_history[:, :, idx]) + \
                                  K_history[:, :, idx+1] * \
                                  (Z_history[idx+1] - np.dot(H, x_hat_history[:, :, idx]))
    x_hat_history[:, :, idx+1] = x_hat_history[:, :, idx] + dx_hat_history[:, :, idx+1]*DT

    # Residual & Error
    # dr(t) = dz(t) - H(t)*xhat(t)*dt)
    residual[:, idx+1] = Z_history[idx+1] - \
                         np.dot(H, x_hat_history[:, :, idx+1])
    err_history[:, :, idx+1] = x_hat_history[:, :, idx+1] - x_history[:, :, idx+1]


# ============================================================================
# Plots
# ============================================================================
print("Printing plots...")
if display_plot == 1 and single_run_plot == 1:
    plt.figure(1)
    plt.title('Kalman Gains')
    plt.plot(T, K_history[0, :, :].transpose())
    plt.plot(T, K_history[1, :, :].transpose())
    plt.plot(T, K_history[2, :, :].transpose())
    plt.ylabel('Gain')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    plt.show()


print("Missile State Estimation Done")