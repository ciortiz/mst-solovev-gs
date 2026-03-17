# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:59:32 2026

@author: darkr
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def psi(R, Z, A=1, C=5, mu0 = 4e-7):
    initial_guess = [1, 1]
    Ra, Rb = fsolve(bcs, initial_guess)
    return (C/8)*( (R**2-Ra**2)**2-Rb**4) + (C/2)*(R**2 - (A/C))*Z**2

def P(R, Z):
    return 

def bcs(vars, A = 1, C = 5, R0 = 1, a = 0.5, mu0 = 4e-7):
    Ra, Rb = vars
    eq1 = Ra**2 + Rb**2 - (R0 + a)**2
    eq2 = R0**4 - 2*Ra**2 * R0**2 + Ra**4 - Rb**4 + 4*a**2 * A/C
    return [eq1, eq2]

initial_guess = [1, 1]
Ra, Rb = fsolve(bcs, initial_guess)


R0 = 1
a = 0.5
R_range = np.linspace(R0-a, R0+a, 1000)
Z_range = np.linspace(-a, a, 1000)

R, Z = np.meshgrid(R_range, Z_range)
psival = psi(R, Z)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.plot(R_range, np.sqrt(0.5**2-Z_range**2), color='Black', lw = 3)
ax.plot(R_range, -np.sqrt(0.5**2-Z_range**2), color='Black', lw = 3)
ax.plot(R0, 0, color='red', marker = 'o')

bar = ax.contourf(R_range, Z_range, psival)
fig.colorbar(bar)
