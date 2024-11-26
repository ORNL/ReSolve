namespace ReSolve 
{
  namespace tests
  {
    /**
     * @brief Class for compting and updating residual.
     * 
     */
    class AxEqualsRhsProblem
    {
      public:

        AxEqualsRhsProblem(const std::string& matrix_filepath, const std::string& rhs_filepath);
        ~AxEqualsRhsProblem();

        ReSolve::matrix::Csr* getMatrix() const;
        ReSolve::vector::Vector* getVector() const;
        ReSolve::vector::Vector* getRhs() const;

        void updateProblem(const std::string& matrix_filepath, 
                           const std::string& rhs_filepath);


      private:

        ReSolve::matrix::Csr* A_;
        ReSolve::vector::Vector* vec_x_;
        ReSolve::vector::Vector* vec_rhs_;
    };

    /**
     * @brief Class with utility functions for verifying test results.
     * 
     */
    class FunctionalityTestHelper
    {
      public:

        FunctionalityTestHelper(ReSolve::real_type tol_init,
                                workspace_type &workspace_init,
                                AxEqualsRhsProblem &problem);

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

        void calculateNorms(AxEqualsRhsProblem &problem);

        real_type calculateSolutionVectorNorm(ReSolve::vector::Vector& vec_x);

        real_type calculateRhsVectorNorm(ReSolve::vector::Vector& vec_x);

        real_type calculateResidualNorm(ReSolve::vector::Vector& vec_r);

        real_type calculateDiffNorm(ReSolve::vector::Vector& vec_x);

        int checkResidualNorm();

        real_type calculateTrueNorm(ReSolve::matrix::Csr& A, ReSolve::vector::Vector& vec_rhs);

        void printNorms(std::string &testname);

        ReSolve::vector::Vector generateResidualVector(ReSolve::matrix::Csr& A,
                                                       ReSolve::vector::Vector& vec_x,
                                                       ReSolve::vector::Vector& vec_rhs);

      private:
        workspace_type& workspace_;
        std::unique_ptr<ReSolve::MatrixHandler> mh_{nullptr};
        std::unique_ptr<ReSolve::VectorHandler> vh_{nullptr};
        ReSolve::real_type tol_{constants::DEFAULT_TOL};
        real_type residual_norm_{-1.0};
        real_type rhs_norm_{-1.0};
        real_type diff_norm_{-1.0};
        real_type norm_solution_vector_{-1.0};
        real_type true_residual_norm_{-1.0};
    };
  } // namespace tests
} // namespace ReSolve
