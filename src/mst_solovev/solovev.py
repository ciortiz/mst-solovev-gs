"""
Objects relevant to the forward and inverse problems of the Grad-Shafranov
equation with Solovev forms.
"""

import numpy as np


def psi(R, Z, C, A, gamma, R_a, R_b):
    """
    Solovev flux function truncated at third order in the basis set of solutions
    to the homogeneous equation.

    Args
    ----
    R, Z (floats or arrays):
        Radial and axial coordinates in a cylindrical system.
    C, A (floats):
        Solovev constants for p' and FF', respectively.
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


def B_Z(R, Z, C, A, gamma, R_a, R_b):
    """
    Axial component of the magnetic field.

    Args
    ----
    R, Z (floats or arrays):
        Radial and axial coordinates in a cylindrical system.
    C, A (floats):
        Solovev constants for p' and FF', respectively.
    gamma, R_a, R_b (floats):
        Undetermined constants associated with the basis functions that solve
        the homogeneous equation.

    Returns
    -------
    float or array (dependent on R and Z types).
    """
    term_1 = (C * gamma / 2.) * (R**2 - R_a**2)
    term_2 = C * (1 - gamma) * Z**2
    return term_1 + term_2


def B_phi(R, Z, C, A, gamma, R_a, R_b):
    """
    Toroidal component of the magnetic field.

    Args
    ----
    R, Z (floats or arrays):
        Radial and axial coordinates in a cylindrical system.
    C, A (floats):
        Solovev constants for p' and FF', respectively.
    gamma, R_a, R_b (floats):
        Undetermined constants associated with the basis functions that solve
        the homogeneous equation.

    Returns
    -------
    float or array (dependent on R and Z types).
    """
    F_0 = 0.26  # T m; for standard, 0.13-T MST-tokamak plasmas
    arg = F_0**2 - (A * C * gamma / 4.) * ((R**2 - R_a**2)**2 - R_b**4)
    return np.sqrt(arg) / R


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
    Solovev constants for FF' and p', respectively.

    Args
    ----
    gamma (float):
        Elongation parameter.
    R_0, a (floats):
        Major and minor radii of the torus.

    Returns
    -------
    float:
        Ratio A/C.
    """
    return (2. * gamma - 1.) * R_0**2 - (gamma / 4.) * a**2


def construct_equilibrium(A, R_0, a):
    """
    Obtain all prescribed and unknown constants needed to construct the Solovev
    flux function.

    Args
    ----
    A (float):
        Solovev constant for FF'.
    R_0, a (floats):
        Major and minor radii of the torus.

    Returns
    -------
    tuple:
        floats: C, gamma, R_a and R_b, in that order.
    """
    gamma, R_a, R_b = solve_coefficients(R_0, a)
    C = A / solve_solovev_ratio(gamma, R_0, a)
    return C, gamma, R_a, R_b


def construct_psi(A, R_0, a):
    """
    Build a Solovev psi(R, Z) from the toroidal geometry and prescribed Solovev
    parameters.

    Args
    ----
    A (float):
        Solovev constant for FF'.
    R_0, a (floats):
        Major and minor radii of the torus.

    Returns
    -------
    psi_func (callable):
        Reference to a Solovev flux function that requires only radial and axial
        input parameters.
    """
    C, gamma, R_a, R_b = construct_equilibrium(A, R_0, a)

    def psi_func(R, Z):
        return psi(R, Z, C, A, gamma, R_a, R_b)

    return psi_func
