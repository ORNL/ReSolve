/**
 * @file Doxygen.hpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-12-07
 * 
 * @copyright Copyright (c) 2023
 * 
 * @mainpage ReSolve Source Code Documentation
 * 
 * ReSolve is a library of GPU-resident linear solvers. It contains iterative
 * and direct linear solvers designed to run on NVIDIA and AMD GPUs, as well as
 * on CPU devices.
 * 
 * @section Name
 * 
 * @section History
 * 
 * The development of Re::Solve sparse linear solver library started as a part
 * of Stochastic Grid Dynamics at Exascale (ExaSGD) project. The overarching
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
 * @section Code Design and Organization
 * 
 */