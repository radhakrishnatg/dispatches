"""
This file contains functions for computing the dimensionless
numbers, friction factor, etc.
"""
from math import log, sqrt, pi


def reynolds(vel, dh, rho, mu):
    return rho * dh * vel / mu


def prandtl(mu, cp, k):
    return mu * cp / k


def cross_sectional_area(diameter):
    return pi * diameter ** 2 / 4


def clamond(Re, eps, d):
    """Calculates Darcy friction factor using a solution accurate to almost
    machine precision. Recommended very strongly. For details of the algorithm,
    see [1]_.

    Parameters
    ----------
    Re : Reynolds number, [-]
    eps : Relative roughness, [m]
    d : diameter of channel, [m]
    Returns
    -------
    fd : Darcy friction factor [-]
    Notes
    -----
    This is a highly optimized function, 4 times faster than the solution using
    the LambertW function, and faster than many other approximations which are
    much less accurate.
    The code used here is only slightly modified than that in [1]_, for further
    performance improvements.

    References
    ----------
    .. [1] Clamond, Didier. "Efficient Resolution of the Colebrook Equation."
       Industrial & Engineering Chemistry Research 48, no. 7 (April 1, 2009):
       3665-71. doi:10.1021/ie801626g.
       http://math.unice.fr/%7Edidierc/DidPublis/ICR_2009.pdf
    """
    X1 = (eps / d) * Re * 0.1239681863354175460160858261654858382699  # (log(10)/18.574).evalf(40)
    X2 = log(Re) - 0.7793974884556819406441139701653776731705  # log(log(10)/5.02).evalf(40)
    F = X2 - 0.2
    X1F = X1 + F
    X1F1 = 1.0 + X1F

    E = (log(X1F) - 0.2) / X1F1
    F = F - (X1F1 + 0.5 * E) * E * X1F / (X1F1 + E * (1. + E / 3.0))

    X1F = X1 + F
    X1F1 = 1. + X1F
    E = (log(X1F) + F - X2) / X1F1
    F = F - (X1F1 + 0.5 * E) * E * X1F / (X1F1 + E * (1. + E / 3.0))

    return 1.325474527619599502640416597148504422899 / (F * F)


def generate_clamond_plots():
    re = []


def haaland(Re, eps, d):
    return 1 / (-1.8 * log(((eps / d) / 3.7) ** 1.11 + 6.9 / Re)) ** 2


def turbulent_Churchill_Zajic(Re, Pr, fd):

    Pr_T = 0.85 + 0.015 / Pr
    Nu_di = Re * (fd / 8.0) / (1. + 145 * (8.0 / fd) ** (-1.25))
    Nu_dinf = 0.07343 * Re * (Pr / Pr_T) ** (1.0 / 3.0) * (fd / 8.0) ** 0.5
    return 1.0 / (Pr_T / Pr / Nu_di + (1.0 - (Pr_T / Pr) ** (2 / 3.0)) / Nu_dinf)


def smooth_switch(x, x_low, x_high, y_low, y_high, switch_rate):
    """
    Smooth the transition between two functions y_1 and y_2. The function y
    will be equal to y_1 when x is less than x_1. When x is between x_1 and
    x_2 the function y will transition smoothly and continuously from y_1 to
    y_2. When x is greater than x_2, y will be equal to y_2.
    The higher the switch rate, the more rapidly the switch will transition.
    A value of 10-1000 for switch rate is common.
    """
    switch_point = (x_low + x_high) / 2  # Find the mid point between the x_1 and x_2 bounds
    smoother = x - switch_point  # Adjust x so that the midpoint of x now becomes 0
    switch_limit = (x_high - x_low) / switch_rate
    # Smooth absolute value of mass flow rate
    smoother_abs = (smoother ** 2 + switch_limit ** 2) ** (1/2)
    # Smooth step functions for energy flow rate during flow reversal
    smoother_step_2 = (1 + smoother / smoother_abs) / 2
    smoother_step_1 = (1 - smoother / smoother_abs) / 2
    return smoother_step_2 * y_high + smoother_step_1 * y_low


def conv_htc(Re, mu, cp, k, ff, d):
    Pr = prandtl(mu, cp, k)
    Nu_lam = 48 / 11
    Nu_turb = turbulent_Churchill_Zajic(Re, Pr, ff)
    Nu = smooth_switch(Re, 2030, 2050, Nu_lam, Nu_turb, 5)
    return Nu * k / d


def u_tes(r, k, a, b):
    zz = r + ((a ** 3 * (4 * b ** 2 - a ** 2) + a * b ** 4 * (4 * log(b / a) - 3)) / (4 * k * (b ** 2 - a ** 2) ** 2))
    return 1 / zz


def out_diam(fa, a):
    return sqrt(fa / pi + a ** 2)
