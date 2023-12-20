/**
 * @file Doxygen.hpp
 * @author Slaven Peles (peless@ornl.gov)
 * @brief 
 * 
 * @mainpage ReSolve Source Code Documentation v1.0
 * 
 * ReSolve is a library of GPU-resident linear solvers. It contains iterative
 * and direct linear solvers designed to run on NVIDIA and AMD GPUs, as well as
 * on CPU devices. This is the main page of source code documentation intended
 * for developers who want to contribute to ReSolve code. General documentation
 * is available at <a href=https://resolve.readthedocs.io>readthedocs</a>.
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
 * @subsection solvers_subsec Solvers
 * 
 * @subsection matvecs_subsec Matrix and Vector Classes
 * 
 * @subsection handlers_subsec Matrix and Vector Handlers
 * 
 * @subsection workspaces_subsec Workspaces
 * 
 * @subsection backends_subsec Hardware Backends
 * 
 * @subsection utils_subsec Utilities
 * 
 */
