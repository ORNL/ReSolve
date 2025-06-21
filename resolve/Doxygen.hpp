/**
 * @file Doxygen.hpp
 * @author Slaven Peles (peless@ornl.gov)
 * @brief
 *
 * @mainpage ReSolve Source Code Documentation
 *
 * ReSolve is a library of GPU-resident linear solvers. It contains iterative
 * and direct linear solvers designed to run on NVIDIA and AMD GPUs, as well as
 * on CPU devices. This is the main page of source code documentation intended
 * for developers who want to contribute to ReSolve code. General documentation
 * is available at <a href=https://resolve.readthedocs.io>readthedocs</a>. The
 * ReSolve project is hosted on <a href=https://github.com/ORNL/ReSolve>GitHub</a>.
 *
 *
 * @section name_sec Name
 *
 * Linear solvers are typically used within an application where a series of
 * systems with same sparsity pattern is solved one after another, such as in
 * the case of dynamic simulations or optimization. An efficient linear solver
 * design will _re-solve_ systems with the same sparsity pattern while reusing
 * symbolic operations and memory allocations from the prior systems, therefore
 * the name ReSolve.
 *
 * @section history_sec History
 *
 * The development of Re::Solve sparse linear solver library started as a part
 * of Stochastic Grid Dynamics at Exascale
 * (<a href="https://www.exascaleproject.org/research-project/exasgd/">ExaSGD</a>)
 * subproject of the Exascale Computing Project
 * (<a href="https://www.exascaleproject.org/">ECP</a>). The overarching
 * goal was to address project’s major technology gap at the time. The project
 * required a fast sparse direct solver that is (natively) GPU-resident and at
 * the same time, highly optimized, without unnecessary memory traffic nor
 * repeated memory allocations and deallocations. A stable iterative refinement
 * strategy was needed too, as the systems solved in ExaSGD were ill-conditioned
 * by construction. During the project, it turned out that combining different
 * codes and strategies (e.g., using LU decomposition from one solver library
 * and following it with an alternative triangular solve and iterative
 * refinement) is a winning approach and that a flexible library that
 * facilitates this type of free mix-and-match solver style was needed. At the
 * same time, the technology gap is not unique to ExaSGD, and affects many
 * other applications whose overall performance heavily depends on the
 * performance of the linear solver. Hence, Re::Solve was developed to be a
 * versatile solver library that is flexible (e.g., same iterative solvers can
 * be used for iterative refinement and direct solvers as preconditioners),
 * designed to solve a sequence of linear systems with the same sparsity
 * pattern without unnecessary recomputation and re-allocations, easy to
 * integrate with applications, and capable of running on both AMD and NVIDIA
 * GPUs.
 *
 * @section design_sec Code Design and Organization
 *
 * Re::Solve is designed to be portable so it can run on different hardware;
 * extensible so that new solvers can be added easily, reusing existing
 * ifrastructure; and configurable so that different solving strategies can
 * be instantiated with existing solvers classes. Re::Solve operates on its
 * matrix and vector objects, which can be instantiated as standalone objects
 * or as wrappers for user provided data. Sparse BLAS operations are
 * implemented in MatrixHandler and VectorHandler classes. The workspace
 * classes store data that is reusable over several numerical linear algebra
 * operations. The main Re::Solve functionality is implemented in solver
 * classes. Re::Solve has three hardware backends -- the CPU backend and
 * backends to devices supporting CUDA and HIP languages, respectively.
 *
 * @subsection solvers_subsec Solvers
 *
 * Linear solver classes contain iterative (F)GMRES solvers with low-sync
 * Gram-Schmidt orthogonalization and a variant with randomized Krylov
 * subspace. Direct solvers are implemented using third party matrix
 * (re)factorization and triangular solver functions. The SystemSolver class
 * enables combining these classes into a user specified custom solver.
 *
 * @subsection matvecs_subsec Matrix and Vector Classes
 *
 * Linear solvers operate on Re::Solve's matrix and vector classes.
 * Functionality of these classes is minimal -- they manage memory allocation,
 * initialization, and copying matrix and vector data. They can be also used
 * as wrappers for existing user provided matrix and vector data.
 *
 * @subsection handlers_subsec Matrix and Vector Handlers
 *
 * Sparse BLAS operations are implemented in MatrixHandler and VectorHandler
 * classes rather than as standalone functions. The handlers manage access to
 * workspaces that are needed for many sparse linear algebra functions,
 * especially in GPUs implementations. This desing allows for reuse of the
 * workspace and computation results when the linear solver is called to solve
 * a sequence of similar linear systems.
 *
 * Each linear solver object has a dedicated matrix and a vector handler.
 *
 * @subsection workspaces_subsec Workspaces
 *
 * Workspace classes store data buffers and handles to different third party
 * objects, such as CUDA SDK and ROCm functions. Currently implemented are
 * LinAlgWorkspaceCUDA, LinAlgWorkspaceHip, and LinAlgWorkspaceCpu.
 *
 * Each linear solver object owns one dedicated workspace, which is shared
 * with a matrix and a vector handler belonging to the same solver.
 *
 * @subsection backends_subsec Hardware Backends
 *
 * In addition to sequential implementation of compute kernels, Re::Solve
 * has implementations in HIP and CUDA languages.
 *
 * @subsection utils_subsec Utilities
 *
 * Re::Solve implements several utility classes for managing input and
 * output.
 *
 */
