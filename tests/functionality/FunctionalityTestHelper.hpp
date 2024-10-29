namespace ReSolve 
{
  namespace tests
  {
    // Captain! Rename all these!
    class AxEqualsRhsProblem
    {
      public:

        AxEqualsRhsProblem(std::string& matrix_filepath, std::string& rhs_filepath);
        ~AxEqualsRhsProblem();

        ReSolve::matrix::Csr* getMatrix();
        ReSolve::vector::Vector* getVector();
        ReSolve::vector::Vector* getRhs();

        void updateProblem(std::string& matrix_filepath, 
                           std::string& rhs_filepath);


      private:

        ReSolve::matrix::Csr* A_;
        ReSolve::vector::Vector* vec_x_;
        ReSolve::vector::Vector* vec_rhs_;
    };

    class FunctionalityTestHelper
    {
      public:

        FunctionalityTestHelper(ReSolve::real_type tol_init,
                                workspace_type &workspace_init );

        ~FunctionalityTestHelper();

        int checkResult(ReSolve::matrix::Csr& A,
                                      ReSolve::vector::Vector& vec_rhs,
                                      ReSolve::vector::Vector& vec_x,
                                      ReSolve::SystemSolver& solver,
                                      std::string testname);

        void printIterativeSolverStats(SystemSolver& solver);

        int checkNormOfScaledResiduals(ReSolve::matrix::Csr& A,
                                      ReSolve::vector::Vector& vec_rhs,
                                      ReSolve::vector::Vector& vec_x,
                                      ReSolve::SystemSolver& solver);

        int checkRelativeResidualNorm(ReSolve::vector::Vector& vec_rhs,
                                      ReSolve::vector::Vector& vec_x,
                                      ReSolve::SystemSolver& solver);

        void calculateNorms( ReSolve::matrix::Csr& A,
                             ReSolve::vector::Vector& vec_rhs,
                             ReSolve::vector::Vector& vec_x );

        real_type calculateSolutionVectorNorm(ReSolve::vector::Vector& vec_x);

        real_type calculateRhsVectorNorm(ReSolve::vector::Vector& vec_x);

        real_type calculateResidualNorm(ReSolve::vector::Vector& vec_r);

        real_type calculateDiffNorm(ReSolve::matrix::Csr& A, ReSolve::vector::Vector& vec_x);

        int checkResidualNorm();

        real_type 
        calculateTrueNorm(ReSolve::matrix::Csr& A, ReSolve::vector::Vector& vec_rhs);

        void printNorms(std::string &testname);

        ReSolve::vector::Vector 
        generateResidualVector(ReSolve::matrix::Csr& A,
                                  ReSolve::vector::Vector& vec_x,
                                  ReSolve::vector::Vector& vec_rhs);

      private:
        workspace_type& workspace_;
        ReSolve::MatrixHandler* mh_{nullptr};
        ReSolve::VectorHandler* vh_{nullptr};
        ReSolve::real_type tol_{constants::DEFAULT_TOL};
        real_type residual_norm_{-1.0};
        real_type rhs_norm_{-1.0};
        real_type diff_norm_{-1.0};
        real_type norm_solution_vector_{-1.0};
        real_type true_residual_norm_{-1.0};
    };
  } // namespace tests
} // namespace ReSolve
