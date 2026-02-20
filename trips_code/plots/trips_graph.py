import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.io as spio
from scipy import sparse as spsparse

def plot_speedup(A_file, rhs_file, x_true, name):
    # Load the data.
    A = spio.mmread(A_file)
    b = spio.mmread(rhs_file)
    x_true = spio.mmread(x_true)

    # Create a figure for throughput curves
    fig, ax = plt.subplots(figsize=(9, 8))
        
    ax.plot(x_true, "-r", label = 'x_true')
    ax.plot(b, label = "Re::Solve Solver", alpha = 0.5)
    #ax.plot(ReSolve_GMRES(A, A.T, b, 45, x_0 = np.zeros((len(b), 1)), b_iter = 0, stopping_rule = "dp"), label = "Hybrid BA GMRES (Givens)")

    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., fontsize=15)
    
    # Set the axes bounding box to a 9:8 vertical-to-horizontal ratio.
    ax.set_box_aspect(9/8)
    
    fig.tight_layout()
    fig.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def ReSolve_GMRES(A, B, b, n_iter, x_0, b_iter = 0, stopping_rule = "dp"):

    # Q History (V in ReSolve)
    Q = np.zeros((len(b), n_iter + 1))

    # Hessenberg Matrix
    H = np.zeros((n_iter + 1, n_iter))

    # Norm of b
    b = B @ b
    norm_b = np.linalg.norm(b)

    # First residual (Using x_0 = 0)
    residual = b
    Q[:, 0] = (residual/np.linalg.norm(residual)).flatten()

    # Initialize residual history
    rs_hist = np.zeros(n_iter + 1)
    rs_hist[0] = np.linalg.norm(residual)

    # Residual Norm History
    rs_norm_hist = np.zeros(n_iter + 1)
    rs_norm_hist[0] = np.linalg.norm(residual)

    m, n = A.shape

    # x history
    X = np.zeros((n, n_iter))

    # c and s history
    c = np.zeros(n_iter)
    s = np.zeros(n_iter)

    gammas = np.zeros(n_iter)

    k = 0
    while (k < n_iter):
        # Form the next basis vector
        v = B @ (A @ Q[:,k])

        # Orthogonalize new vector against the basis
        for j in range(k + 1):
            H[j, k] = Q[:, j].conj().T @ v
            v = v - H[j, k] * Q[:, j]
        
        
        # Add vector norm to hessenberg matrix
        H[k+1, k] = np.linalg.norm(v)
        Q[:, k+1] = (v / H[k+1, k]).flatten()

        # Given's Rotation
        if (k != 0):
            for i in range(1, k + 1):
                #print(i)
                i1 = i - 1
                t = H[i1][k]
                H[i1][k] = c[i1] * t + s[i1] * H[i][k]
                H[i][k] = c[i1] * H[i][k] - s[i1] * t
        
        gamma = np.sqrt(H[k][k] * H[k][k] + H[k+1][k] * H[k+1][k])
        # Next Rotation
        c[k] = H[k][k]/gamma
        s[k] = H[k+1][k]/gamma
        
        h_kk = H[k][k]
        h_k1k = H[k+1][k]

        H[k][k] = c[k] * h_kk + s[k] * h_k1k
        H[k+1][k] = c[k] * h_k1k - s[k] * h_kk

        # Rotation applied to residual history
        rs_hist[k + 1] = -s[k] * rs_hist[k]
        rs_hist[k] = c[k] * rs_hist[k]

        # Add residual norm to history
        rs_norm_hist[k + 1] = np.abs(rs_hist[k + 1])

        k += 1
    
    # Regularization Givens Rotations
    lambdah = 5.046719957351252e-06
    identity_shape = H.shape[1]
    for i in range(H.shape[1]):
        # Obtain rotated elements
        reg_i = H[i][i]
        reg_j = np.sqrt(lambdah)

        gamma = np.sqrt(reg_i * reg_i + reg_j * reg_j)

        # Rotation
        reg_c = reg_i/gamma
        reg_s = reg_j/gamma

        # Rotate all elements in the row
        H[i][i] = reg_c * reg_i + reg_s * reg_j
        # print(reg_c)
        for j in range(i + 1, H.shape[1]):
            H[i][j] = reg_c * H[i][j]
        
        # Rotate the residual vector
        rs_norm_hist[i] = reg_c * rs_norm_hist[i]
        

    # Backsolve for y
    y = np.zeros(n_iter)
    y[n_iter - 1] = rs_hist[n_iter-1]/H[n_iter-1,n_iter-1]
    n_elem = 0
    for ii in range(2, n_iter + 1):
        # Num of elements that must be subtracted from rhs at index n_iter - ii
        n_elem += 1
        t = rs_hist[n_iter - ii]
        for iii in range(n_elem):
            t = t -  H[n_iter - ii, n_iter - 1 -iii] * y[n_iter-1 - iii]
        y[n_iter - 1 - n_elem] = t/H[n_iter-1-n_elem, n_iter-1-n_elem]

    y_actual = np.linalg.solve(H[:n_iter, :n_iter], rs_hist[:n_iter])
    x = (Q[:,:n_iter] @ y)

    return x

def main():
    if len(sys.argv) < 2:
        print("Usage: trips_graph.py <rhs> <x_true> <plot_name>")
        sys.exit(1)

    A_file = sys.argv[1]
    rhs_file = sys.argv[2]
    x_true = sys.argv[3]
    name = sys.argv[4]
    plot_speedup(A_file, rhs_file, x_true, name)

if __name__ == "__main__":
    main()