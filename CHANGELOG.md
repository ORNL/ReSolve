# Re::Solve Changelog

## HyKKT Release changes

- Added classes and tests for permutation, Ruiz scaling, Cholesky factorization, and matrix multiplication and addition.

## Changes to Re::Solve since release 0.99.2

- Added cmake-format

- The MatrixHandler::scaleAddI function was added to calculate A := cA + I

- The MatrixHandler::scaleAddB function was added to calculate A := cA + B

## Changes to Re::Solve in release 0.99.2

### Major Features

1. Re::Solve now works reliably with asymmetric matrices, with no need for intermediate CSC storage. 
This requires switching $L$ with $U$ and $P$ with $Q$ and reinterpretting them as CSR instead of CSC. 
It is seamless from the user perspective and fixed many bugs.

2. Significant improvements to documentation and instructions inside and outside the code. Added general API description, including details on memory space synchronization.

3. Added more rigorous checks for PRs for clang formatting and to compile without warnings and memory leaks.

4. Updated pull request and issue templates.

### Bug Fixes

1. Fixed a bug that produced inaccurate results for some asymmetric matrices with major feature 1.

2. Synchronized devices after HIP functions. HIP executes asynchronously, so bugs occured without synchronization.

3. Corrected the way cmake finds suitsparse.

4. Fixed various memory leaks and compiler warnings.

### Minor Features and Enhancements

1. Changed all examples and tests to use Csr format, added uniform command line parsers (no longer hard-coded), and decluttered them.

2. Added asymmetric matrices and well-conditioned matrices to the test suite.

3. Removed RocSparse "fast mode" triangular solver and use RocSolver triangular solver only as it is now faster and removes dependencies.

4. Put sorting inside the KLU extraction because many solvers assume sorted factors and there's no need to reimplement sorting constantly.

5. Removed duplicate code, added code comments, corrected code to fit guidelines, removed magic numbers, and simplified code where possible.

6. Added the ability to reset a workspace without completely destroying it.

7. Improved testing and added tests where they were missing.

8. Added kernels for multiplying a vector by a diagonal matrix and a general matrix by a diagonal matrix (left and right).

9. Prohibitted sloppy memory syncing and added more descriptive error messages when a prohibited action is attempted.

10. The code now tracks the updated status for each vector in a multivector.

11. Added the ability to reuse a transpose allocation.

12. Added the ability to generically set solver parameters.

13. Added LUSOL direct solver, which can factorize matrices and extract factors independently of KLU.

14. Various Spack updates.
