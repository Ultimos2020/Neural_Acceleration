# sigmoid_numpy.py

import numpy as np

def sigmoid(x):
    """Compute the sigmoid of x."""
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    """Compute the sigmoid derivative of x."""
    return  np.exp(-x)/ (1.0 + np.exp(-x))**2

def main():
    start = -15.0   # lower bound of x
    end   =  15.0   # upper bound of x
    step  =  0.25   # step size

    # Generate x values from start to end (inclusive)
    x_values = np.arange(start, end + step, step)

    # Calculate and print sigmoid for each x
    for x in x_values:
        y = sigmoid(x)
        z = sigmoid_derivative(x)
        print(f"x = {x:6.2f} → sigmoid(x) = {y:.6f} → sigmoid'(x) = {z:.6f}")
        #print(f"x = {x:6.2f} → sigmoid(x) = {y:.6f}")

if __name__ == "__main__":
    main()

