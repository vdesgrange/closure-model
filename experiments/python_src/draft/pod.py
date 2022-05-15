import numpy as np
from scipy import linalg as la
from heat import get_heat

def pod(U, t_n, x_n, k_min=0, k_max=1):
    # Get covariance matrix
    C = np.matmul(U.T, U) # / X

    # Solve eigenproblem
    K, v = la.eigh(C, b=None)

    # Sort eigen values and vectors
    indexes = np.argsort(-abs(K))
    K = K[indexes] # eigenvalue
    v = v[:, indexes] # eigenvector = spatial mode

    # Compute time coefficient
    A = np.matmul(U, v.T);

    u_tilde = np.zeros((t_n, x_n))

    for k in range(k_min, k_max, 1):
        Vk = v[k, :]
        Vk = Vk[np.newaxis, :]
        Ak = A[:, k] 
        Ak = Ak[:, np.newaxis]
        u_tilde_k = np.matmul(Ak, Vk)
        u_tilde += u_tilde_k

    return u_tilde


if __name__ == '__main__':
    U, _ = get_heat(1.0, 0.0, 1.0, 0.0, 20, 100)
    U_pod = pod(U, 20, 100, 0, 35)
