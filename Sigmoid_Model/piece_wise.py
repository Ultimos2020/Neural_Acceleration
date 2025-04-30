# sigmoid_numpy.py

import numpy as np
import pandas as pd
import csv

max = -100000000
max_x = 0
min = 100000000
min_x = 0

start = -10.0   # lower bound of x
end   =  10.0   # upper bound of x
step  =  0.00001   # step size


def sigmoid(x):
    """Compute the sigmoid of x."""
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    """Compute the sigmoid derivative of x."""
    return  np.exp(-x)/ (1.0 + np.exp(-x))**2

def sigmoid_derivative_2(x):
    """Compute the sigmoid second derivative of x."""
    return np.exp(-x) * (np.exp(-x)-1) / (1.0 + np.exp(-x))**3

def point_extractor(x):
    """For a symmetry"""
    x4 = -x/2
    x5 = -x
    x6 = -x - x/2
    x1 = -x6
    x2 = -x5
    x3 = -x4
    #print(f"x1 = {x1:.6f}, x2 = {x2:.6f}, x3 = {x3:.6f}, x4 = {x4:.6f}, x5 = {x5:.6f}, x6 = {x6:.6f}")
    return x1, x2, x3, x4, x5, x6

def extrema(y,x):
    """Find the extrema of the sigmoid function."""
    global max, min, max_x, min_x
    if y > max:
        max = y
        max_x = x
    if y < min:
        min = y
        min_x = x

def piece_wise(x1, x2, x3, x4, x5, x6):
    m0 = 0
    c0 = 0
    #print("x3-x2", x3-x2)
    m1 = (sigmoid(x2)-sigmoid(x1))/(x2-x1)
    c1 = intercept(x1,x2)
    m2 = (sigmoid(x3)-sigmoid(x2))/(x3-x2)
    c2 = intercept(x2,x3)
    m3 = (sigmoid(x4)-sigmoid(x3))/(x4-x3)
    c3 = intercept(x3,x4)
    m4 = (sigmoid(x5)-sigmoid(x4))/(x5-x4)
    c4 = intercept(x4,x5)
    m5 = (sigmoid(x6)-sigmoid(x5))/(x6-x5)
    c5 = intercept(x5,x6)
    m6 = 0
    c6 = 1
    print(f" y = {m0:.6f}*x + {c0:.6f} for x < {x1:.6f}")
    print(f" y = {m1:.6f}*x + {c1:.6f} for {x1:.6f} <= x < {x2:.6f}")
    print(f" y = {m2:.6f}*x + {c2:.6f} for {x2:.6f} <= x < {x3:.6f}")
    print(f" y = {m3:.6f}*x + {c3:.6f} for {x3:.6f} <= x < {x4:.6f}")
    print(f" y = {m4:.6f}*x + {c4:.6f} for {x4:.6f} <= x < {x5:.6f}")
    print(f" y = {m5:.6f}*x + {c5:.6f} for {x5:.6f} <= x < {x6:.6f}")
    print(f" y = {m6:.6f}*x + {c6:.6f} for x >= {x6:.6f}")
    return m0, c0, m1, c1, m2, c2, m3, c3, m4, c4, m5, c5, m6, c6

rows = []
def error_measure(x1, x2, x3, x_4, x_5, x_6, m0, c0, m1, c1, m2, c2, m3, c3, m4, c4, m5, c5, m6, c6):
    #start = -15.0   # lower bound of x
    #end   =  15.0   # upper bound of x
    #step  =  0.001   # step size
    error_2 = 0
    x_total = 0
    x_values = np.arange(start, end + step, step)
    global max, min, max_x, min_x
    for x in x_values:
        y_true = sigmoid(x)
        if x < x1:
            y = m0*x + c0
        elif x1 <= x < x2:
            y = m1*x + c1
        elif x2 <= x < x3:
            y = m2*x + c2
        elif x3 <= x < x_4:
            y = m3*x + c3
        elif x_4 <= x < x_5:
            y = m4*x + c4
        elif x_5 <= x < x_6:
            y = m5*x + c5
        else:
            y = m6*x + c6
        error = y_true - y
        error = abs(error)
        extrema(error,x)
        error_2 = error ** 2 + error_2
        x_total = x_total + x
        #print(f"x = {x:6.2f} → sigmoid(x) = {y_true:.6f} → piecewise(x) = {y:.6f} → error = {error:.6f}")
        rows.append([x, y_true, y, error])
    
    df = pd.DataFrame(rows, columns=['x', 'sigmoid(x)', 'piecewise(x)', 'error'])
    df.to_csv('sigmoid_piecewise.csv', index=False, float_format='%.6f')
    
    error_squared = error_2/len(x_values)
    print(f"error_squared = {error_squared:.6f}")





def intercept (x1,x2):

    start_local = x1
    end_local = x2
    step_local = 0.001
    x_values = np.arange(start_local, end_local + step_local, step_local)
    x_total = 0
    c = 0
    #print("x1 = ", x1)
    #print("x2 = ", x2)
    for x in x_values:
        y = sigmoid(x)
        y_der = sigmoid_derivative(x)
        temp = y - x*y_der
        c = temp + c
        x_total = x_total + x
    #print ("x_total = ", x_total)
    c = c/len(x_values)
    return c

def main():

    global max, min, max_x, min_x
    # Generate x values from start to end (inclusive)
    x_values = np.arange(start, end + step, step)

    # Calculate and print sigmoid for each x
    for x in x_values:
        y = sigmoid(x)
        z = sigmoid_derivative(x)
        z2 = sigmoid_derivative_2(x)
        extrema(z2,x)
        #print(f"x = {x:6.2f} → sigmoid(x) = {y:.6f} → sigmoid'(x) = {z:.6f} -> sigmoid''(x) = {z2:.6f}")
    #print(f"max = {max:.6f} at x = {max_x:.6f}")
    #print(f"min = {min:.6f} at x = {min_x:.6f}")
    x1, x2, x3, x4, x5, x6 = point_extractor(max_x)
    max = -100000000
    max_x = 0
    min = 100000000
    min_x = 0
   
    m0, c0, m1, c1, m2, c2, m3, c3, m4, c4, m5, c5, m6, c6 = piece_wise(x1, x2, x3, x4, x5, x6)
    error_measure(x1, x2, x3, x4, x5, x6, m0, c0, m1, c1, m2, c2, m3, c3, m4, c4, m5, c5, m6, c6)


    print(f"max = {max:.6f} at x = {max_x:.6f}")
    print(f"min = {min:.6f} at x = {min_x:.6f}")

if __name__ == "__main__":
    main()

