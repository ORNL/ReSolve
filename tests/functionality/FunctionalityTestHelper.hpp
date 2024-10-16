// Design requirement: minimize number of #include in the header

// we are doing this because CUDA is a very clingy dependency and will invade this
// declarative space unless we use forward declarations combined with CPP/source includes
// CUDA deps handled at the linking stage instead of the building stage


//#include <resolve/Common.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
// #include <resolve/SystemSolver.hpp>

// #if defined (RESOLVE_USE_CUDA)
// #include <resolve/LinSolverDirectCuSolverRf.hpp>
//   using workspace_type = ReSolve::LinAlgWorkspaceCUDA;
//   std::string memory_space("cuda");
// #elif defined (RESOLVE_USE_HIP)
// #include <resolve/LinSolverDirectRocSolverRf.hpp>
//   using workspace_type = ReSolve::LinAlgWorkspaceHIP;
//   std::string memory_space("hip");
// #else
//   using workspace_type = ReSolve::LinAlgWorkspaceCpu;
//   std::string memory_space("cpu");
// #endif

namespace ReSolve {
  namespace tests{
    class FunctionalityTestHelper
    {
      public:

        FunctionalityTestHelper( ReSolve::real_type tol_init = constants::DEFAULT_TOL );

        ~FunctionalityTestHelper();

        int checkRefactorizationResult(ReSolve::matrix::Csr& A,
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

      private:
        workspace_type workspace_;
        ReSolve::MatrixHandler* mh_{nullptr};
        ReSolve::VectorHandler* vh_{nullptr};
        ReSolve::real_type tol_{constants::DEFAULT_TOL};
        real_type residual_norm_{-1.0};
        real_type rhs_norm_{-1.0};
        real_type diff_norm_{-1.0};
        real_type true_norm_{-1.0};
        real_type true_residual_norm_{-1.0};
    };
  } // namespace tests
} // namespace ReSolve
