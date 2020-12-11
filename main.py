import numpy as np

def sweep_method(arr, r):
    """
    Sweep method for solving boundary value problem
    
    :param arr: 2D array of linear system
    :param r: right part of linear system
    :return: y0, ..., y[n-1]
    """
    n = len(r)
    b, c, d = np.zeros(n), np.diag(arr), np.zeros(n)
    for i in range(n-1):
        b[i+1] = arr[i+1, i]
        d[i] = arr[i, i+1]
    
    delta, lmbd = np.zeros(n), np.zeros(n)
    for j in range(n):
        delta[j] = -d[j]/(c[j]+b[j] * delta[j-1])
        lmbd[j] = (r[j]-b[j] * lmbd[j-1]) / (c[j] + b[j] * delta[j-1])
    
    y = np.zeros(n)
    for k in range(n-1, -1, -1):
        y[k] = lmbd[k] + delta[k] * y[k-n+1]
    return y

if __name__ == "__main__":
    arr = np.array([[19, -20, 0, 0, 0, 0], [280.145202, -554.756, 275.4103, 0, 0, 0], [0, 280.067, -554.756, 275.4884, 0, 0], [0, 0, 279.994, -554.756, 275.5615, 0], [0, 0, 0, 279.925, -554.756, 275.63001], [0, 0, 0, 0, 1, -1]])
    r = np.array([-2, 1.76, 1.82, 1.88, 1.94, -0.06])
    #arr = np.array([[-11, 12, 0, 0], [1825/18, -199.2, 1775/18, 0], [0, 1925/19, -199.2, 1875/19], [0, 0, -10, 10]])
    #r = np.array([2, 1.8, 1.9, 1])
    y = sweep_method(arr, r)
    print(y)
