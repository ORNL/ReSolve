//this is to solve the system, can call different linear solvers if necessary
namespace ReSolve
{
  class LinSolverDirectKLU;
  class LinSolverDirect;
  class LinSolverIterative;
  class GramSchmidt;
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

      // enum ResidualNormType {RR = 0, NSR};

      SystemSolver();
      SystemSolver(LinAlgWorkspaceCUDA* workspaceCuda, 
                   std::string factor = "klu",
                   std::string refactor = "cusolverrf",
                   std::string solve = "cusolverrf",
                   std::string ir = "none");
      SystemSolver(LinAlgWorkspaceHIP*  workspaceHip, 
                   std::string factor = "klu",
                   std::string refactor = "rocsolverrf",
                   std::string solve = "rocsolverrf",
                   std::string ir = "none");

      ~SystemSolver();

      int initialize();
      int setMatrix(matrix::Sparse* A);
      int analyze(); //    symbolic part
      int factorize(); //  numeric part
      int refactorize();
      int refactorizationSetup();
      int precondition();
      int preconditionerSetup();
      int solve(vector_type*  rhs, vector_type* x); // for triangular solve
      int refine(vector_type* rhs, vector_type* x); // for iterative refinement

      // we update the matrix once it changed
      int updateMatrix(std::string format, int* ia, int* ja, double* a);

      LinSolverDirect& getFactorizationSolver();
      LinSolverDirect& getRefactorizationSolver();
      LinSolverIterative& getIterativeSolver();

      real_type getVectorNorm(vector_type* rhs);
      real_type getResidualNorm(vector_type* rhs, vector_type* x);
      real_type getNormOfScaledResiduals(vector_type* rhs, vector_type* x);

      // Get solver parameters
      const std::string getFactorizationMethod() const;
      const std::string getRefactorizationMethod() const;
      const std::string getSolveMethod() const;
      const std::string getRefinementMethod() const;
      const std::string getOrthogonalizationMethod() const;

      // Set solver parameters
      void setFactorizationMethod(std::string method);
      void setRefactorizationMethod(std::string method);
      void setSolveMethod(std::string method);
      void setRefinementMethod(std::string method, std::string gs = "cgs2");

    private:
      LinSolverDirect* factorizationSolver_{nullptr};
      LinSolverDirect* refactorizationSolver_{nullptr};
      LinSolverIterative* iterativeSolver_{nullptr};
      GramSchmidt* gs_{nullptr};

      LinAlgWorkspaceCUDA* workspaceCuda_{nullptr};
      LinAlgWorkspaceHIP*  workspaceHip_{nullptr};

      MatrixHandler* matrixHandler_{nullptr};
      VectorHandler* vectorHandler_{nullptr};

      bool isSolveOnDevice_{false};

      matrix_type* L_{nullptr};
      matrix_type* U_{nullptr};

      index_type* P_{nullptr};
      index_type* Q_{nullptr};

      vector_type* resVector_{nullptr};
    
      matrix::Sparse* A_{nullptr};

      // Configuration parameters
      std::string factorizationMethod_;
      std::string refactorizationMethod_;
      std::string solveMethod_;
      std::string irMethod_;
      std::string gsMethod_;

      std::string memspace_;
  };
} // namespace ReSolve
