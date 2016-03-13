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

print("Missile State Estimation Done")