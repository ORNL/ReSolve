//this is to solve the system, can call different linear solvers if necessary
namespace ReSolve
{
  class LinSolverDirectKLU;

  namespace vector
  {
    class Vector;
  }

  namespace matrix
  {
    class Sparse;
  }

  class SystemSolver
  {
    public:
      using vector_type = vector::Vector;

      SystemSolver();
      SystemSolver(std::string factorizationMethod, std::string refactorizationMethod, std::string solveMethod, std::string IRMethod);

      ~SystemSolver();

      int setMatrix(matrix::Sparse* A);
      int analyze(); //    symbolic part
      int factorize(); //  numeric part
      int refactorize();
      int solve(vector_type* x, vector_type* rhs); // for triangular solve
      int refine(vector_type* x, vector_type* rhs); // for iterative refinement

      // we update the matrix once it changed
      int updateMatrix(std::string format, int * ia, int *ja, double *a);

    private:
    
      matrix::Sparse* A_{nullptr};
      std::string factorizationMethod;
      std::string refactorizationMethod;
      std::string solveMethod;
      std::string IRMethod;

      int setup();
      //internal function to setup the different solvers. IT IS RUN ONCE THROUGH CONSTRUCTOR.

      // add factorizationSolver, iterativeSolver, triangularSolver
      ReSolve::LinSolverDirectKLU* KLU_{nullptr};

  };
} // namespace ReSolve
