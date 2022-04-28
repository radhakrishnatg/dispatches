from idaes.core.util.math import smooth_abs
from pyomo.environ import Var
import matplotlib.pyplot as plt


def get_slopes_intercepts(data):
    """
    This function determines the slopes and intercepts of the
    piecewise linear function passing through the points in data.
    For a given set of points, this function returns a piecewise
    linear function of the form:
    f(x) = alpha[0] + beta[0] * x, for -inf <= x <= x1,
           alpha[1] + beta[1] * x, for x1 <= x <= x2,
           alpha[2] + beta[2] * x, for x2 <= x <= x3,
           ...
           alpha[n] + beta[n] * x, for xn <= x <= inf

    We assume alpha[0] = alpha[1] and beta[0] = beta[1], and
    alpha[n] = alpha[n - 1] and beta[n] = beta[n - 1].

    Args:
        data: list of points [(x1, y1), (x2, y2),...,(xn, yn)]
    Returns:
        alpha: Dictionary of intercepts
        beta: Dictionary of slopes
    """

    # Line passing through points (x1, y1) and (x2, y2) is given by
    # (y - y1) = ((y2 - y1) / (x2 - x1)) (x - x1). Arranging it in
    #  standard form gives y = alpha + beta * x, where
    # alpha = y1 - ((y2 - y1) / (x2 - x1)) x1, and
    # beta = ((y2 - y1) / (x2 - x1))

    alpha = {}
    beta = {}

    for i in range(len(data) - 1):
        x1, y1 = data[i]
        x2, y2 = data[i + 1]
        alpha[i + 1] = y1 - ((y2 - y1) / (x2 - x1)) * x1
        beta[i + 1] = ((y2 - y1) / (x2 - x1))

    alpha[0] = alpha[1]
    beta[0] = beta[1]
    alpha[len(data)] = alpha[len(data) - 1]
    beta[len(data)] = beta[len(data) - 1]

    return alpha, beta


def eval_piecewise_function(alpha, beta, brk_pts, x):
    """
    The function returns the value of the piecewise function
    evaluated at point x
    Args:
        alpha: intercepts
        beta: slopes
        brk_pts: set of break points
        x: point at which function needs to be evaluated
    Returns:
        f(x)
    """
    if x <= brk_pts[0][0]:
        seg = 0
    elif x > brk_pts[-1][0]:
        seg = len(brk_pts)
    else:
        for i in range(len(brk_pts) - 1):
            if brk_pts[i][0] < x <= brk_pts[i + 1][0]:
                seg = i + 1

    return alpha[seg] + beta[seg] * x


def get_break_points(data_train, tol, brk_pts=None, generate_plots=False):
    """
    This function determines the set of points needed from the training
    set (data_train) such that the maximum absolute value of the
    error between data_train and its piecewise linear approximation
    is less than tol everywhere in the domain of interest.
    Args:
        data_train: Set of points/data for training
        tol: Maximum deviation allowed
        brk_pts: Set of points which must be included
        generate_plots: Generates plots if True
    Returns:
        brk_pts: Points needed to construct the piecewise linear approximation
    """
    if brk_pts is None:
        # If no data is provided, then start with the first and the last points
        brk_pts = [data_train[0], data_train[-1]]
    else:
        # Check whether the first and the last points are included
        # If not, add them to the set
        if not data_train[0] in brk_pts:
            brk_pts.insert(0, data_train[0])
        if not data_train[-1] in brk_pts:
            brk_pts.insert(len(brk_pts), data_train[-1])

    x = [data_train[i][0] for i in range(len(data_train))]
    y_true = [data_train[i][1] for i in range(len(data_train))]

    flag = True

    while flag:
        alpha, beta = get_slopes_intercepts(brk_pts)
        y_approx = [eval_piecewise_function(alpha, beta, brk_pts, x[i])
                    for i in range(len(x))]
        residual = [abs(y_true[i] - y_approx[i]) for i in range(len(x))]

        if max(residual) > tol:
            max_violation_point = residual.index(max(residual))
            brk_pts.append(data_train[max_violation_point])
            brk_pts.sort()

        else:
            flag = False

    if generate_plots:
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(x, y_true, color="blue", label="True Data")
        ax1.plot(x, y_approx, color="red", label="surrogate")
        ax1.set_title(f"Valid for {brk_pts[0][0]} <= x <= {brk_pts[-1][0]}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("f(x)")
        ax1.legend()

        ax2.plot(x, residual, color="blue")
        ax2.set_title("Residuals")
        ax2.set_xlabel("x")
        ax2.set_ylabel("Residual")
        fig.tight_layout()
        plt.show()

    return brk_pts


def smooth_piecewise_linear(data, x):
    """
    This function returns a smoothened piecewise linear function
    passing through the points in data.
    Args:
        data: list of points [(x1, y1), (x2, y2), ..., (xn, yn)]
        x: Pyomo variable
    Returns:
        f(x): smoothened piecewise linear function
    """

    if not isinstance(x, Var):
        raise TypeError("x must be a pyomo Var object")

    # Get slopes and intercepts of the piecewise linear function
    alpha, beta = get_slopes_intercepts(data)

    # The piecewise linear function can be expressed as follows:
    # f(x) = a0 + a1 * x + \sum_{k=1}^n (0.5 * delta[k] * |x - xk|), where
    # a0 = alpha[0] - \sum_{k=1}^n (0.5 * delta[k])
    # a1 = 0.5 * (beta[0] + beta[n])
    # delta[k] = beta[k] - beta[k - 1]
    # Then, we replace |x - xk| with the smooth_abs function

    delta = {k: beta[k] - beta[k - 1] for k in range(1, len(data) + 1)}
    a0 = alpha[0] - sum(0.5 * delta[k] * data[k - 1][0] for k in delta)
    a1 = 0.5 * (beta[0] + beta[len(data)])

    fx = a0 + a1 * x
    for k in delta:
        fx += 0.5 * delta[k] * smooth_abs(x - data[k - 1][0])

    return fx


# if __name__ == '__main__':
#     from pyomo.environ import ConcreteModel, Expression
#     import pandas as pd
#
#     # For discharge
#     HT_data = pd.read_csv('HT_data.csv')
#     x = HT_data["enth"].to_list()
#     y_true = HT_data["temp"].to_list()
#     ht_dat = [(x[i], y_true[i]) for i in range(len(x))]
#     brk_pts = get_break_points(ht_dat, tol=3)
#
#     m = ConcreteModel()
#     m.x = Var()
#     m.y = Expression(expr=smooth_piecewise_linear(data=brk_pts, x=m.x))
#
#     y_approx = []
#     for i in x:
#         m.x.set_value(i)
#         y_approx.append(m.y.expr())
#
#     residual = [y_true[i] - y_approx[i] for i in range(len(y_true))]
#
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#
#     ax1.plot(x, y_true, color="blue", label="True Data")
#     ax1.plot(x, y_approx, color="red", label="surrogate")
#     ax1.set_title(f"Valid for {brk_pts[0][0]} <= x <= {brk_pts[-1][0]}")
#     ax1.legend()
#
#     ax2.plot(x, residual, color="blue")
#     ax2.set_title("Residuals")
#     plt.show()
