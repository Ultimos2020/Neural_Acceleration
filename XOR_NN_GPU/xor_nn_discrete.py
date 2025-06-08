def target_function(x):
    return x[0] ^ x[1]

def MSE(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def rectilinear(x):
    return x * (x > 0)

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [target_function(i) for i in x]