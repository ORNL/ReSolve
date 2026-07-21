# Re::Solve Changelog

## HyKKT Release changes

- Added classes and tests for permutation, Ruiz scaling, Cholesky factorization, Schur complement conjugate gradient and matrix multiplication and addition.

- Changed random number generation int tests to be C++ style and fixed-seed, to avoid random failures.

- Added Schur Complement Conjugate Gradient class.

## Changes to Re::Solve since release 0.99.2

- Added cmake-format.

- Reworked templates to include example tests.

- Removed unnecessary full facotorization in the examples.

- Update method names in `VectorHandler` to match BLAS naming.

- Added `cons` counterparts to `Vector::getData` methods.

- Made Vector::copyToExternal able to copy from device to host and vice versa.

- Added `diagSolve`, `max`, and `abs` vector operations.

- Added `allocateWithExternalSparsityPattern` for sparse matrices.

- Added initial guess support for FGMRES and randomized FGMRES iterative solvers.

- Improved coding guidelines for developers on floating point conventions.

- Made spack builds more robust.

- Removed unnecessary device synchronization and added where needed (HIP only calls).

- Changed variable and function names to be more explanatory.

- Added Developer Guide documentation for vector and matrix classes and handlers.

- Remove automatic memory allocation and memory syncing in various memory management functions and require the user to explicitly manage their own memory.

- Set Re::Solve's real type and matrix/vector index type at CMake configuration time.

- Added size assertions to `VectorHandler::gemv` and fixed vector sizing in `GramSchmidt` accordingly.

- Added line to testlog to tell the user to expect warnings as part of normal testing.

- Fixed GPU refactorization allocation and synchronization errors, uninitialized GMRES example initial guesses, and CUDA ILU0 failures on stored numerical zero pivots.

- Optimized CGS2 coefficient accumulation in `GramSchmidt` by eliminating an unnecessary device-to-host synchronization.

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

15. Added examples/sysGmres.cpp to demonstrate how to use SystemSolver with GMRES. 

16. Updated MatrixHandler::addConst to return integer error codes instead of void.

17. Added a preconditioner interface class so users can define their own preconditioners.

18. Added left preconditioning support for GMRES and a user-defined preconditioner class.

19. Added optional timing output to `gpuRefactor`, `kluRefactor`, `gluRefactor`, and `sysRefactor` plus benchmark utilities for parsing logs and generating timing/residual plots.
