//this is to solve the system, can call different linear solvers if necessary
namespace ReSolve
{
  class LinSolverDirectKLU;
  class LinSolverDirect;
  class LinAlgWorkspaceCUDA;
  class LinAlgWorkspaceHIP;
  class MatrixHandler;
  class VectorHandler;

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
      using matrix_type = matrix::Sparse;

      SystemSolver();
      SystemSolver(LinAlgWorkspaceCUDA* workspaceCuda);
      SystemSolver(LinAlgWorkspaceHIP*  workspaceHip);
      SystemSolver(std::string factorizationMethod, std::string refactorizationMethod, std::string solveMethod, std::string IRMethod);

      ~SystemSolver();

      int initialize();
      int setMatrix(matrix::Sparse* A);
      int analyze(); //    symbolic part
      int factorize(); //  numeric part
      int factorize_setup(); //  numeric part
      int refactorize();
      int refactorize_setup();
      int solve(vector_type* x, vector_type* rhs); // for triangular solve
      int refine(vector_type* x, vector_type* rhs); // for iterative refinement

      int setCudaWorkspace(LinAlgWorkspaceCUDA* workspaceCuda);

      // we update the matrix once it changed
      int updateMatrix(std::string format, int* ia, int* ja, double* a);

      void setFactorizationMethod(std::string method);
      void setRefactorizationMethod(std::string method);
      void setSolveMethod(std::string method);
      void setIterativeRefinement(std::string method);

    private:
    
      matrix::Sparse* A_{nullptr};
      std::string factorizationMethod_;
      std::string refactorizationMethod_;
      std::string solveMethod_;
      std::string irMethod_;
      std::string gsMethod_;

      //internal function to setup the different solvers. IT IS RUN ONCE THROUGH CONSTRUCTOR.

      // add factorizationSolver, iterativeSolver, triangularSolver
      LinSolverDirectKLU* KLU_{nullptr};
      LinSolverDirect* refactorSolver_{nullptr};

      LinAlgWorkspaceCUDA* workspaceCuda_{nullptr};
      LinAlgWorkspaceHIP*  workspaceHip_{nullptr};

      MatrixHandler* matrixHandler_{nullptr};
      VectorHandler* vectorHandler_{nullptr};

      bool isSolveOnDevice_{false};

      matrix_type* L_{nullptr};
      matrix_type* U_{nullptr};

      index_type* P_{nullptr};
      index_type* Q_{nullptr};

      vector_type* dummy_{nullptr};
  };
} // namespace ReSolve
