# HyKKT

## Description
HyKKT (pronounced as _hiked_) is a package for solving systems of equations and unknowns resulting from an iterative solution of an optimization
problem, for example optimal power flow, which uses hardware accelerators (GPUs) efficiently.

The HyKKT package contains a linear solver tailored for Karush Kuhn Tucker (KKT) linear systems and
deployment on hardware accelerator hardware such as GPUs. The solver requires
all blocks of the $4\times 4$ block system: 

```math
\begin{bmatrix}
    H + D_x     & 0         & J_c^T     & J_d^T \\
      0         & D_s       & 0           & -I  \\
     J_c        & 0         & 0           & 0   \\
     J_d        & -I        & 0           & 0
\end{bmatrix}
\begin{bmatrix}
  \Delta x \\ \Delta s \\ \Delta y_c \\ \Delta y_d
\end{bmatrix} =
\begin{bmatrix}
  \tilde{r}_x \\ r_s \\ r_c \\ r_d
\end{bmatrix}
```

separately and solves the system to a desired
numerical precision exactly via block reduction and conjugate gradient on the
schur complement. Please see the [HyKKT paper](https://www.tandfonline.com/doi/abs/10.1080/10556788.2022.2124990) for mathematical details.

## Integration within ReSolve

HyKKT code within ReSolve is not yet fully integrated. The code is currently experimental, but is actively in development.
The main issue is the requirement that the user pass matrix and vector blocks, rather than the constructed KKT system. 
This is currently not available within ReSolve as it is specific to HyKKT and KKT matrices.
TODO:
- Work with AMD to fix bug in Cholesky factorization.
- Allow the representation of a matrix or vector by blocks within ReSolve.
- Schur Compliment Conjugate Gradient within HyKKT.
- Complete HyKKT integration within ReSolve.


## Dependencies

Most HyKKT dependencies are identical to ReSolve. Exceptions are:
- HyKKT does not require KLU
- For AMD builds, HyKKT requires HIP/ROCm >= 6.4 due to a known bug in SpGEMM.
