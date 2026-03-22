"""
Solve the forward problem of the Grad-Shafranov equation with Solovev forms.
"""

import matplotlib.pyplot as plt
import numpy as np

from mst_solovev.geometry import R_0, a 
from mst_solovev.solovev import construct_psi


C = -0.02  # T / m^2; Solovev constant for p
psi_fn = construct_psi(C, R_0, a)


### Define coordinate plane
scale = 1.1  # padding scalar
R_arr = np.linspace(R_0 - scale * a, R_0 + scale * a, 1000)
Z_arr = np.linspace(-scale * a, scale * a, 1000)
R, Z = np.meshgrid(R_arr, Z_arr)


### Define landmarks
psi_val = psi_fn(R, Z)
R_shell = np.linspace(R_0 - a, R_0 + a, 1000)
Z_shell = np.sqrt(np.maximum(0., a**2 - (R_shell - R_0)**2))


### Plotting
fig, ax = plt.subplots(figsize = (10, 8))

flux_contours = ax.contourf(R, Z, psi_val, levels = 30)  # heatmap of contours
ax.contour(R, Z, psi_val, levels = 20, linewidths = 0.7)  # flux surfaces
ax.contour(R, Z, psi_val, levels = [0], colors = 'pink', linewidths = 2)  # psi=0 contour

ax.plot(R_shell, Z_shell, color = 'grey', linewidth = 2)  # upper conductive shell
ax.plot(R_shell, -Z_shell, color = 'grey', linewidth = 2)  # lower conductive shell

ax.plot(R_0, 0, marker = 'o', color = 'black')  # cross-section geometric center

ax.set_xlabel(r'$R$ (m)')
ax.set_ylabel(r'$Z$ (m)')
ax.set_aspect('equal')

fig.colorbar(flux_contours, ax = ax, label = r'$\Psi(R,Z)$')

#plt.savefig('forward_solution.pdf', dpi = 300)
plt.show()
