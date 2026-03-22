"""
Objects relevant to the forward and inverse problems of the Grad-Shafranov
equation with Solovev forms.
"""

import numpy as np


def psi(R, Z, C, A, gamma, R_a, R_b):
    """
    Solovev flux function.

    Args
    ----
    R, Z (floats or arrays):
        Radial and axial coordinates in a cylindrical system.
    C, A (floats):
        Solovev constants for p and F, respectively.
    gamma, R_a, R_b (floats):
        Undetermined constants associated with the basis functions that solve
        the homogeneous equation.

    Returns
    -------
    float or array (dependent on R and Z types).
    """
    term_1 = (C * gamma / 8.) * ((R**2 - R_a**2)**2 - R_b**4)
    term_2 = (C / 2.) * ((1. - gamma) * R**2 + (A / C)) * Z**2
    return term_1 + term_2


def solve_coefficients(R_0, a):
    """
    Determine the coefficients of the basis functions using boundary conditions.

    Args
    ----
    R_0, a (floats):
        Major and minor radii of the torus.

    Returns
    -------
    tuple:
        floats: gamma, R_a and R_b, in that order.
    """
    gamma = 1.  # dimensionless; elongation is 1 for circular cross-section
    R_a = np.sqrt(R_0**2 + a**2)
    R_b = np.sqrt(2. * R_0 * a)

    return gamma, R_a, R_b


def solve_solovev_ratio(gamma, R_0, a):
    """
    Determine the ratio A/C using boundary conditions, where A and C are the
    Solovev constants for F and p, respectively.

    Args
    ----
    gamma (float):
        Elongation parameter (dimensionless).
    R_0, a (floats):
        Major and minor radii of the torus.

    Returns
    -------
    float:
        Ratio A/C.
    """
    return (2. * gamma - 1.) * R_0**2 - (gamma / 4.) * a**2


def construct_psi(C, R_0, a):
    """
    Build a Solovev psi(R, Z) from the toroidal geometry and prescribed Solovev
    parameters.

    Args
    ----
    C (float):
        Solovev constant for p.
    R_0, a (floats):
        Major and minor radii of the torus.

    Returns
    -------
    psi_func (callable):
        Reference to a Solovev flux function that requires only radial and axial
        input parameters.
    """
    gamma, R_a, R_b = solve_coefficients(R_0, a)
    A = C * solve_solovev_ratio(gamma, R_0, a)

    def psi_func(R, Z):
        return psi(R, Z, C, A, gamma, R_a, R_b)

    return psi_func
