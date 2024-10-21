namespace ReSolve 
{
  namespace tests
  {
    class FunctionalityTestHelper
    {
      public:
        FunctionalityTestHelper(ReSolve::real_type tol_init = constants::DEFAULT_TOL);
        ~FunctionalityTestHelper();

        int checkResultNorms(ReSolve::matrix::Csr& A,
                                      ReSolve::vector::Vector& vec_rhs,
                                      ReSolve::vector::Vector& vec_x,
                                      ReSolve::SystemSolver& solver,
                                      std::string testname);

        void printIterativeSolverStats(SystemSolver& solver);

        int checkNormOfScaledResiduals(ReSolve::matrix::Csr& A,
                                      ReSolve::vector::Vector& vec_rhs,
                                      ReSolve::vector::Vector& vec_x,
                                      ReSolve::vector::Vector& vec_r,
                                      ReSolve::SystemSolver& solver);

        int checkRelativeResidualNorm(ReSolve::vector::Vector& vec_rhs,
                                      ReSolve::vector::Vector& vec_x,
                                      const real_type residual_norm,
                                      const real_type rhs_norm,
                                      ReSolve::SystemSolver& solver);

        real_type calculate_solution_vector_norm( ReSolve::vector::Vector& vec_x );

        real_type calculate_rhs_vector_norm( ReSolve::vector::Vector& vec_x );

        real_type calculate_residual_norm( ReSolve::vector::Vector& vec_r );

        ReSolve::vector::Vector 
        generate_residual_vector( ReSolve::matrix::Csr& A,
                                  ReSolve::vector::Vector& vec_x,
                                  ReSolve::vector::Vector& vec_rhs );

      private:
        workspace_type workspace_;
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
