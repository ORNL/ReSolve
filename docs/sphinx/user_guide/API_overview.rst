Re::Solve General API Description 
=====================================

Solver Subroutines
------------------

For direct solvers, the following functions are provided and must be called in this order:

* ``analyze()`` - Performs symbolic factorization, currently only with KLU on the CPU.
* ``factorize()`` - Performs numeric factorization, given a symbolic factorization.
* ``refactorizationSetup()`` - Prepares for a new numeric factorization, given a new matrix with the same sparsity structure (required only if using ``refactorize()``).
* ``solve()`` - Solves the linear system, given a numeric factorization.
* ``refactorize()`` - Reuses the symbolic factorization to perform a new numeric factorization, given a new matrix.

Subsequent systems only require calls to ``refactorize()`` and ``solve()`` if the sparsity structure hasn't changed.

Note some examples do not reuse the factorization from the 0-th iteration because the numeric sparsity structure changes and that's how the matrices are stored. In a practical application, the sparsity structure is likely to remain the same. If it does not, it is the user's responsibility to reallocate memory and call ``analyze()`` and ``factorize()`` again. The user should enforce that symbolic nonzeros are stored explicitly, even if they are numerically zero.

Function Requirements
---------------------

Functions must be used as in the examples and tests:

* Workspaces are required for GPU solvers. The generic workspace_type is required for backend agnostic code.
* Handles are created with the ``initializeHandles()`` function and destroyed with the default destructor.
* Allocate memory first, preferably with the ``ReSolve::MatrixHandler`` or ``ReSolve::VectorHandler`` classes.
* Memory must be allocated before attempting to copy to it.
* Solver output variables must be allocated before passing to the solver.
* Use setup functions to initialize objects, where available. Do not call them more than once.
* Call the set or factor functions before the corresponding reset or refactor functions.  
* Deallocate memory at the end. 

Functions in ReSolve that take pointers as arguments will have the following requirements:

* Pointers must be valid and point to allocated memory, unless the function's use is to allocate memory.
* Pointers that are not marked as const must point to memory that is writable.
* Pointers that are marked as const must point to memory that is readable.

Memory Synchronization
----------------------

* For CPU solvers, memory is always in sync.
* For GPU solvers, the user must synchronize memory manually. 
* Manually call ``setUpdated()`` if you modify the contents of a ``ReSolve::MatrixHandler`` or ``ReSolve::VectorHandler`` object without using the object's member functions.
* Manually call ``syncData(memspace)`` when you modify the other memory space and want them to match. Both memory spaces must be allocated.