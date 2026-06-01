Vector and Matrix Classes and Handlers
======================================

Background
----------

The purpose of this page is to help new developers understand how Re::Solve
separates data containers, operation handlers, backend workspaces, and memory
spaces. In particular, it explains the difference between vector and matrix
classes and the ``VectorHandler`` and ``MatrixHandler`` classes that operate on
them.

The main distinction is that vector and matrix classes store data, while
handler classes perform operations on that data. This distinction is important
when writing code that needs to port to different backends run (e.g. CPU, CUDA, and HIP).

This separation allows solver logic to remain independent of backend-specific
vector and matrix operations.

The main questions this page is meant to answer are:

* What object stores the data?
* Where does the data live?
* What object performs the operation?
* What backend resources does the operation need?
* What is the difference between a vector or matrix class and a vector or
  matrix handler?
* What needs to happen when data is loaded on the host but used on the device?

This page is not meant to document every method in detail. It is meant to give
a practical mental model for reading and writing backend-capable Re::Solve code.

Core Design
-----------

The main design idea is that Re::Solve separates storage, operations, and
backend resources.

The major pieces are:

* ``vector::Vector`` objects store vector data.
* Matrix objects, such as ``matrix::Csr``, store sparse matrix data.
* ``VectorHandler`` objects perform vector operations.
* ``MatrixHandler`` objects perform matrix operations.
* ``LinAlgWorkspace`` objects provide backend-specific resources for the
  handlers.
* ``memory::HOST`` describes data stored in host-accessible memory.
* ``memory::DEVICE`` describes data stored in device-accessible memory.

This means that a vector or matrix object is not automatically a CPU or GPU
operation. The data object stores the values. The handler performs the
operation. The workspace gives the handler the backend resources it needs.

This separation helps the same solver path run with different backend
implementations.

Vector Objects
--------------

A ``vector::Vector`` object represents vector data. The vector object is a data
container. It stores the size of the vector and the data associated with that
vector. Before a vector is used, it must be allocated in a memory space.

Simplified example:

.. code:: cpp

   vector::Vector* x = new vector::Vector(n);
   x->allocate(memory::HOST);

If the vector is loaded or initialized on the host but later used by a GPU
backend, the data may need to be synchronized to the device.

Simplified example:

.. code:: cpp

   if (memspace == memory::DEVICE)
   {
     x->syncData(memory::DEVICE);
   }

The important distinction is that allocation and operation are separate steps.
Allocating the vector controls where the data is stored. Calling a handler
method controls what operation is performed on the data.

This is useful because the same vector object may be part of a CPU test path or
a GPU test path, depending on how it is allocated, synchronized, and passed to
backend-specific operations.

Matrix Objects
--------------

Matrix objects represent matrix data. Like vector objects, matrix objects are
data containers. They store or describe the matrix data, but they do not perform
matrix operations by themselves.

Sparse matrices are commonly stored in compressed sparse formats, such as CSR
(compressed sparse row) and CSC (compressed sparse column). These formats store
only the nonzero values of a sparse matrix along with index information that
describes where those values belong.

In Re::Solve, matrix objects such as ``matrix::Csr`` store sparse matrix data.
A CSR matrix stores the matrix dimensions, nonzero count, and sparse matrix
data. Like vectors, a matrix object must be allocated in a memory space before
it is used.

Simplified example:

.. code:: cpp

   matrix::Csr* A = new matrix::Csr(num_rows, num_cols, nnz);
   A->allocateMatrixData(memory::HOST);

In file-loading paths, matrix data may need to be loaded into host memory
first. For example, Matrix Market file readers write into host-accessible
memory. If the test is running on a GPU backend, the matrix can then be
synchronized to device memory.

SCCG test path example:

.. code:: cpp

   matrix::Csr* h = new matrix::Csr(2278, 2278, 11304, true, false);
   h->allocateMatrixData(memory::HOST);
   io::updateMatrixFromFile(h_file, h);

   if (memspace_ == memory::DEVICE)
   {
     h->syncData(memory::DEVICE);
   }

This pattern matters because the memory space used for loading data is not
always the same as the memory space used for computation.

Vector Handlers
---------------

A ``VectorHandler`` performs operations on ``vector::Vector`` objects. It does not replace
the vector class. Instead, it provides backend-specific operations
that act on existing vector data.

A ``VectorHandler`` may perform operations such as:

* ``dot``
* ``scal``
* ``axpy``

A useful way to think about the difference is:

* ``vector::Vector`` stores the vector data.
* ``VectorHandler`` performs vector operations on that data.

A useful way to separate the roles is that ``vector::Vector`` stores the data, while ``VectorHandler`` performs operations on that data.
For example, a vector object may hold the entries of a residual vector, while a
vector handler may compute a dot product, scale the vector, or add one vector
to another.

Matrix Handlers
---------------

A ``MatrixHandler`` performs operations on matrix objects such as ``matrix::Csr``. It does not replace
the matrix class. Instead, it provides backend-specific matrix operations that
act on existing matrix data.

A ``MatrixHandler`` may perform operations such as:

* ``matvec``
* ``transpose``

A useful way to separate the roles is that ``matrix::Csr`` stores the data, while ``MatrixHandler`` performs matrix operations on that data.
For example, a matrix object may hold the CSR representation of a sparse
matrix, while a matrix handler may perform a sparse matrix-vector product or
construct a transpose.

Handler Setup
-------------

Handlers are created using a workspace for the selected backend. A simplified
setup pattern is:

.. code:: cpp

   WorkspaceType workspace;
   workspace.initializeHandles();

   MatrixHandler matrix_handler(&workspace);
   VectorHandler vector_handler(&workspace);

The handler uses the workspace that was created for the selected backend. This
is why backend-capable solver code should generally receive the correct
handlers from the caller instead of creating a hard-coded CPU, CUDA, or HIP
handler internally.

Workspaces
----------

Workspace classes provide the backend-specific resources needed by handlers. A
CPU workspace, CUDA workspace, and HIP workspace may initialize different
backend handles or library resources.

The general setup is:

1. Create the workspace for the selected backend.
2. Initialize the workspace handles.
3. Create matrix and vector handlers using that workspace.
4. Pass those handlers into the solver or test fixture.

Simplified SCCG setup example:

.. code:: cpp

   WorkspaceType workspace;
   workspace.initializeHandles();

   MatrixHandler matrix_handler(&workspace);
   VectorHandler vector_handler(&workspace);

   HykktSchurComplementConjugateGradientTests test(memspace,
                                                   matrix_handler,
                                                   vector_handler);

This keeps the solver or test fixture from being tied to only one backend.

Principle of Operation
----------------------

The basic flow for backend-capable code is:

1. Create or load vector and matrix data.
2. Allocate that data in the correct memory space.
3. If data is loaded on the host and used on the device, synchronize it to the
   device.
4. Create the backend workspace.
5. Create handlers from that workspace.
6. Pass the handlers into the solver or test path.
7. Use the handlers to perform vector and matrix operations.

This flow keeps the data, operation, and backend setup separate. It also makes
it easier to identify whether a problem is caused by data storage, memory
movement, backend setup, or the solver algorithm itself.

Re::Solve Context
-----------------

Re::Solve examples are designed around repeated linear solver use cases. The
public Re::Solve documentation describes examples that emulate a nonlinear
solver calling the linear solver repeatedly. This matters because repeated
solver calls can make setup cost, memory movement, and backend resource
management important.

The public HyKKT documentation describes HyKKT as a solver for
Karush-Kuhn-Tucker systems that can use hardware accelerators efficiently. The
HyKKT description also explains that the solver uses block reduction and
conjugate gradient on the Schur complement.

This background is useful for understanding why the SCCG path needs careful
handling of matrix dimensions, memory spaces, and backend-specific handlers.

SCCG Example
------------

SCCG stands for Schur Complement Conjugate Gradient. The SCCG test path is a
useful example because it uses vector objects, matrix objects, vector handlers,
matrix handlers, workspaces, and memory spaces together.

In the SCCG test path, the matrices are represented with ``matrix::Csr``
objects. This makes SCCG a useful example of how data containers and operation
handlers work together in a backend-capable solver path.

SCCG uses a Schur complement structure. In the test path, the matrices do not
all have the same dimensions, and this is expected.

The main matrices are:

* ``H``: a square matrix used in the inner solve.
* ``Jc``: a rectangular matrix.
* ``Jc_tr``: the transpose of ``Jc``.

A simplified operation chain is:

1. Multiply by ``Jc_tr``.
2. Solve with ``H``.
3. Multiply by ``Jc``.

Because of this structure, not every temporary vector has the same size. Some
vectors match the outer system dimension. Other vectors match the inner solve
dimension. The important requirement is that each matrix and vector matches the
operation being performed.

This is similar to other system designs where each component has a specific
role. The matrix dimensions, memory spaces, and handlers all need to match the
part of the solver path where they are being used.

Important Implementation Detail
-------------------------------

One important detail in the SCCG test path is that the Matrix Market file
readers write into host-accessible memory. This means the test data should be
loaded into ``memory::HOST`` first.

For GPU backends, the data should then be synchronized to ``memory::DEVICE``.
This avoids trying to load file data directly into device memory when the file
reader expects host-accessible memory.

The pattern is:

1. Allocate in ``memory::HOST``.
2. Load the file data.
3. If running on ``memory::DEVICE``, synchronize to device memory.

This applies to both matrix and vector test data.

Why Solver Paths Receive Handlers
---------------------------------

Solver paths that support multiple backends should receive backend-specific
handlers from the caller because the caller knows which backend is being used.
If a solver creates its own handler internally, it can accidentally create a
handler for the wrong backend.

The safer design is:

* The caller or test runner selects the backend.
* The caller or test runner creates the correct workspace.
* The caller or test runner creates the correct matrix and vector handlers.
* The solver receives and uses those handlers.

In the SCCG path, this allows the same solver code to work with CPU, CUDA, and
HIP backends.

Inputs and Outputs
------------------

The main inputs to this code pattern are:

* Matrix and vector data.
* A selected memory space, such as ``memory::HOST`` or ``memory::DEVICE``.
* A backend workspace.
* Matrix and vector handlers.
* Solver-specific data, such as matrix dimensions and solver tolerance.

The main outputs are:

* Correctly allocated and synchronized data.
* Backend-specific matrix and vector operations.
* A solver path that can run on more than one backend.
* A clearer separation between storage, computation, and backend resources.

Common Details to Watch For
---------------------------

The following points may not be clear when first reading this part of the code:

* File readers may require host-accessible memory.
* Loading data and using data may happen in different memory spaces.
* A ``vector::Vector`` or ``matrix::Csr`` object stores data, while a handler performs an
  operation.
* A workspace provides backend-specific resources for handlers.
* A solver that supports multiple backends should receive backend-specific
  handlers from the caller instead of creating a hard-coded backend handler
  internally.
* Rectangular matrices can be expected in SCCG because the Schur complement
  path uses different inner and outer dimensions.
* For GPU tests, loading into ``memory::HOST`` first and then synchronizing to
  ``memory::DEVICE`` may be necessary.
* A test that passes on CPU may still expose memory-space or backend-handler
  issues on CUDA or HIP.

Checklist for Backend-Capable Code
----------------------------------

When writing or reviewing code that should work on CPU and GPU backends, check
the following:

* Is the object allocated before it is used?
* Is the object allocated in the memory space expected by the next operation?
* If data was loaded on the host, is it synchronized to the device before GPU
  operations?
* Are the matrix and vector dimensions consistent with the operation chain?
* Are the handlers created from the correct backend workspace?
* Is the solver receiving backend-specific handlers from the caller?

Suggested Validation
--------------------

When changing code that uses these classes and handlers, it is useful to test
the relevant CPU and GPU paths when the local environment supports them. For an
SCCG-related change, this may include building the CPU and CUDA configurations
and running the SCCG test executable.

Example commands may vary by environment, but the basic checks are:

.. code:: shell

   cmake --build build-cpu
   ./build-cpu/tests/unit/hykkt/hykkt_sccg_test

   cmake --build build-cuda
   ./build-cuda/tests/unit/hykkt/hykkt_sccg_test

System Analysis
---------------

The main purpose of this structure is to make backend-capable solver code
easier to reason about. The vector and matrix classes provide the data storage.
The handlers provide the operations. The workspace provides backend resources.
The memory space describes where the data lives and where operations should
occur.

This separation is especially useful for solver code that needs to work across
CPU, CUDA, and HIP. It reduces the chance that solver code will accidentally
use a CPU-specific handler inside a GPU path. It also makes the memory movement
more explicit when data is loaded on the host and then used on the device.

In the SCCG test path, this structure helps explain why the test loads data
into host memory first, why it synchronizes to device memory for GPU backends,
and why SCCG receives matrix and vector handlers from the caller.

This design also fits the larger Re::Solve and HyKKT motivation. Public ORNL
and Re::Solve materials describe GPU-resident linear solvers as useful in
scientific computing and optimization workflows where linear solves can
dominate runtime. In those workflows, keeping data movement and backend
operations organized is part of making the solver path practical on modern CPU
and GPU systems.

Related Background
------------------

The references below provide additional context for why Re::Solve separates
solver logic, backend operations, and memory movement.

HyKKT is one example of this type of workflow. Shaked Regev's dissertation
describes HyKKT as a method for sparse KKT linear systems that uses an
iterative solver on the Schur complement with an inner Cholesky factorization.
This is relevant to the SCCG path because it explains why matrix-vector
operations, Cholesky solves, matrix dimensions, and backend-specific execution
all appear in the same solver workflow.

Krylov methods provide related background because they are commonly used when
direct methods are too expensive for large systems. Katarzyna Swirydowicz's
dissertation explains repeated large linear solves, Krylov subspace methods,
and GPU implementation tradeoffs for Krylov solvers and preconditioners.

Further Reading
---------------

* `Re::Solve documentation and developer guide <https://resolve.readthedocs.io/en/latest/>`_
* `Re::Solve GitHub repository <https://github.com/ORNL/ReSolve>`_
* `HyKKT GitHub repository <https://github.com/ORNL/hykkt>`_
* `Shaked Regev, Preconditioning Techniques for Sparse Linear Systems <https://purl.stanford.edu/zc485bz0015>`_
* `Katarzyna Swirydowicz, Strategies for Recycling Krylov Subspace Methods and Bilinear Form Estimation <https://vtechworks.lib.vt.edu/items/7aa6a792-b4a1-4bb4-bb45-1e3254bd09e6>`_
* `ORNL publication page on GPU-resident sparse direct linear solvers for ACOPF <https://www.ornl.gov/publication/gpu-resident-sparse-direct-linear-solvers-alternating-current-optimal-power-flow>`_
* `OSTI paper, Iterative Methods in GPU-Resident Linear Solvers for Nonlinear Constrained Optimization <https://www.osti.gov/servlets/purl/2538493>`_
