import numpy as np

x = np.array([[-9, -5, +5],
              [+4, -7, -11],
              [+7, +6, -1],
              [-9, -5, +4],
              [-5, -6, -1],
              [-4, -4, -8],
              [+5, +7, -9],
              [+2, -4, +3],
              [-6, +1, +7],
              [-10, +6, -7]],
             dtype=np.float32)

y_hat = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 1],
                 dtype=np.float32)

W = np.array([-0.1, -0.3, 0.2],
             dtype=np.float32)
b = 2
d = 10
eta = 0.02
ones = np.ones(10)

y = np.matmul(x, W) + b
delta = y - y_hat

W_new = W - eta * np.matmul(x.T, delta) * (1 / d)
b_new = b - eta * np.matmul(ones.T, delta) * (1 / d)

MSE = (1 / (2 * d)) * np.matmul(delta.T, delta)

print("New weights = ", W_new.round(2), " \nNew bias = %.2f \nMSE = %.2f" % (b_new, MSE))
