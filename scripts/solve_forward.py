"""
Solve the forward problem of the Grad-Shafranov equation with Solovev forms.
"""

import matplotlib.pyplot as plt
import numpy as np

from mst_solovev.geometry import R_0, a 
from mst_solovev.solovev import (
        construct_psi,
        construct_equilibrium,
        B_Z,
        B_phi
)


A = 2.  # T; Solovev constant for FF'
psi_fn = construct_psi(A, R_0, a)
C, gamma, R_a, R_b = construct_equilibrium(A, R_0, a)


### Define relevant coordinates
R_flux = np.linspace(R_0 - 1.1 * a, R_0 + 1.1 * a, 1000)
Z_flux = np.linspace(-1.1 * a, 1.1 * a, 1000)
R, Z = np.meshgrid(R_flux, Z_flux)

R_arr = np.linspace(R_0 - a, R_0 + a, 1000)
Z_midplane = np.zeros_like(R_arr)


### Quantities of merit + shell location 
psi_val = psi_fn(R, Z)
Z_shell = np.sqrt(np.maximum(0., a**2 - (R_arr - R_0)**2))

B_Z_midplane = B_Z(R_arr, Z_midplane, C, A, gamma, R_a, R_b)
B_phi_midplane = B_phi(R_arr, Z_midplane, C, A, gamma, R_a, R_b)


### Plot 1: Psi contours
fig1, ax1 = plt.subplots(figsize = (8, 8))

ax1.axhline(0., linestyle = '--', color = 'black', alpha = 0.3)
ax1.axvline(R_0, linestyle = '--', color = 'black', alpha = 0.3)

psi_masked = np.ma.masked_where(psi_val > 0., psi_val)
ax1.contour(
        R, Z, psi_masked,
        levels = 15,
        colors = 'purple',
        linewidths = 1,
        linestyles = 'solid'
)
ax1.contour(  # psi=0 contour
        R, Z, psi_val,
        levels = [0],
        colors = 'red',
        linewidths = 1.5,
        linestyles = 'solid'
)

ax1.plot(  # upper conductive shell
         R_arr, Z_shell,
         color = 'grey',
         linewidth = 2
)
ax1.plot(  # lower conductive shell
        R_arr, -Z_shell,
        color = 'grey',
        linewidth = 2
)

ax1.plot(R_a, 0, marker = 'o', color = 'black')  # magnetic axis 

ax1.set_xlabel(r'$R$ (m)')
ax1.set_ylabel(r'$Z$ (m)')
ax1.set_aspect('equal')

plt.tight_layout()
#plt.savefig('forward_solution.pdf', dpi = 300)
plt.show()


### Plot 2: Midplane B-fields
fig2, (ax2, ax3) = plt.subplots(1, 2, figsize = (10, 6))

ax2.plot(R_arr, B_Z_midplane, color = 'blue', linewidth = 1.5)
ax2.axhline(0., linestyle = 'solid', color = 'black', alpha = 0.3)
ax2.set_xlabel(r'$R$ (m)')
ax2.set_ylabel(r'$B_Z(R, Z=0)$ (T)')
ax2.set_xlim(R_arr[0], R_arr[-1])

ax3.plot(R_arr, B_phi_midplane, color = 'orange', linewidth = 1.5)
ax3.axhline(0., linestyle = 'solid', color = 'black', alpha = 0.3)
ax3.set_xlabel(r'$R$ (m)')
ax3.set_ylabel(r'$B_{\phi}(R, Z=0)$ (T)')
ax3.set_xlim(R_arr[0], R_arr[-1])

plt.tight_layout()
#plt.savefig('midplane_fields.pdf', dpi = 300)
plt.show()
