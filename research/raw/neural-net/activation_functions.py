"""
Activation Functions

These are the activation functions that can be used in the neural network.
"""


# Imports
import numpy as np


def dx(func, x, h=0.0001):
    """
    Calculates the derivative of a function at a given point.

    ### Parameters
    func : `function` - The function to calculate the derivative of.
    x : `float` - The point to calculate the derivative at.
    h : `float` - The step size.

    ### Returns
        `float` - The derivative of the function at the given point.
    """
    return (func(x + h) - func(x - h)) / (2 * h)


def sigmoid(x):
    """

    It is also known as the logistic function

    sigmoid(x) = 1 / (1 + exp(-x))

    Range: [0, 1]

    ### Parameters
    x : `float` - The input value.

    ### Returns
        `float` - The sigmoid value of the input.
    """
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """

    Hyperbolic tangent function.

    tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Range: [-1, 1]

    ### Parameters
    x : `float` - The input value.

    ### Returns
        `float` - The tanh value of the input.
    """
    return np.tanh(x)


def relu(x):
    """

    Rectified Linear Unit function.

    relu(x) = max(0, x)

    Range: [0, inf]

    ### Parameters
    x : `float` - The input value.

    ### Returns
        `float` - The relu value of the input.
    """
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    """

    Leaky Rectified Linear Unit function.

    leaky_relu(x) = max(alpha * x, x)

    Range: (-inf, inf)

    ### Parameters
    x : `float` - The input value.
    alpha : `float` - The slope of the negative part.

    ### Returns
        `float` - The leaky relu value of the input.
    """
    return np.maximum(alpha * x, x)


def elu(x, alpha=0.01):
    """

    Exponential Linear Unit function.

    elu(x) = alpha * (exp(x) - 1) if x < 0 else x

    Range: (-inf, inf)

    ### Parameters
    x : `float` - The input value.
    alpha : `float` - The slope of the negative part.

    ### Returns
        `float` - The elu value of the input.
    """
    return alpha * (np.exp(x) - 1) if x < 0 else x


def softmax(x):
    """
    softmax(x) = exp(x) / sum(exp(x))  \\
    Range: [0, 1]

    ### Parameters
    x : `float` - The input value.

    ### Returns
        `float` - The softmax value of the input.
    """
    return np.exp(x) / np.sum(np.exp(x))


# ? LOSS FUNCTIONS ----------------------------------------------------------------------------------------------- #

def mean_squared_error(y_true, y_pred, derivative: bool = False):
    return y_pred - y_true if derivative else np.mean((y_pred - y_true) ** 2)

