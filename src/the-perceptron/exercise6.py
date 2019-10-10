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
eta = 0.02
ones = np.ones(10)

y = np.matmul(x, W) + b
W_new = W - eta * np.matmul(np.transpose(x), (y - y_hat))
b_new = b - eta * np.matmul(np.transpose(ones), (y - y_hat))

print("New weights = ", W_new, "\nNew bias = ", b_new)
