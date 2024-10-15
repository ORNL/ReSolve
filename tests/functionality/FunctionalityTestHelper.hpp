// Design requirement: minimize number of #include in the header

// we are doing this because CUDA is a very clingy dependency and will invade this
// declarative space unless we use forward declarations combined with CPP/source includes
// CUDA deps handled at the linking stage instead of the building stage


#include <resolve/Common.hpp>
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
        FunctionalityTestHelper( ReSolve::real_type tol_init = constants::DEFAULT_TOL )
          :
          tol_( tol_init )
        {
          workspace_.initializeHandles();
          mh_ = new ReSolve::MatrixHandler(&workspace_);
          vh_ = new ReSolve::VectorHandler(&workspace_);
        }
        ~FunctionalityTestHelper()
        {
          delete mh_;
          delete vh_;
        }
        int checkRefactorizationResult(ReSolve::matrix::Csr& A,
                                      ReSolve::vector::Vector& vec_rhs,
                                      ReSolve::vector::Vector& vec_x,
                                      ReSolve::SystemSolver& solver,
                                      std::string testname)
        {
          using namespace memory;
          using namespace ReSolve::constants;

          int status = 0;
          int error_sum = 0;

          index_type n = A.getNumRows();

          true_norm_ = sqrt(vh_->dot(&vec_x, &vec_x, DEVICE));

          // Allocate vectors
          ReSolve::vector::Vector vec_r(n);
          ReSolve::vector::Vector vec_diff(n);
          ReSolve::vector::Vector vec_test(n);

          // Compute residual norm for the second system
          vec_r.update(vec_rhs.getData(HOST), HOST, DEVICE);
          mh_->setValuesChanged(true, DEVICE);
          status = mh_->matvec(&A, &vec_x, &vec_r, &ONE, &MINUSONE, DEVICE); 
          error_sum += status;
          residual_norm_ = sqrt(vh_->dot(&vec_r, &vec_r, DEVICE));

          //for testing only - control
          rhs_norm_ = sqrt(vh_->dot(&vec_rhs, &vec_rhs, DEVICE));

          // Compute norm of scaled residuals:
          // NSR = ||r||_inf / (||A||_inf * ||x||_inf)
          error_sum += checkNormOfScaledResiduals(A, vec_rhs, vec_x, vec_r, solver);

          //compute ||x_diff|| = ||x - x_true|| norm
          vec_diff.setToConst(1.0, DEVICE);
          vh_->axpy(&MINUSONE, &vec_x, &vec_diff, DEVICE);
          diff_norm_ = sqrt(vh_->dot(&vec_diff, &vec_diff, DEVICE));

          //compute the residual using exact solution
          vec_test.setToConst(1.0, DEVICE);
          vec_r.update(vec_rhs.getData(HOST), HOST, DEVICE);
          status = mh_->matvec(&A, &vec_test, &vec_r, &ONE, &MINUSONE, DEVICE); 
          error_sum += status;
          true_residual_norm_ = sqrt(vh_->dot(&vec_r, &vec_r, DEVICE));

          // Verify relative residual norm computation in SystemSolver
          error_sum += checkRelativeResidualNorm(vec_rhs, vec_x, residual_norm_, rhs_norm_, solver);
        
          std::cout << "Results for " << testname << ":\n\n";
          std::cout << std::scientific << std::setprecision(16);
          std::cout << "\t ||b-A*x||_2                 : " << residual_norm_           << " (residual norm)\n";
          std::cout << "\t ||b-A*x||_2/||b||_2         : " << residual_norm_/rhs_norm_ << " (relative residual norm)\n";
          std::cout << "\t ||x-x_true||_2              : " << diff_norm_               << " (solution error)\n";
          std::cout << "\t ||x-x_true||_2/||x_true||_2 : " << diff_norm_/true_norm_    << " (relative solution error)\n";
          std::cout << "\t ||b-A*x_exact||_2           : " << true_residual_norm_      << " (control; residual norm with exact solution)\n";
          printIterativeSolverStats(solver);

          if (!std::isfinite(residual_norm_/rhs_norm_)) {
            std::cout << "Result is not a finite number!\n";
            error_sum++;
          }
          if (residual_norm_/rhs_norm_ > tol_) {
            std::cout << "Result inaccurate!\n";
            error_sum++;
          }

          return error_sum;
        }

        void printIterativeSolverStats(SystemSolver& solver)
        {
          // Get solver parameters
          real_type tol = solver.getIterativeSolver().getTol();
          index_type restart = solver.getIterativeSolver().getRestart();
          index_type maxit = solver.getIterativeSolver().getMaxit();

          // note: these are the solver's tolerance, different from the testhelper's tolerance

          // Get solver stats
          index_type num_iter   = solver.getIterativeSolver().getNumIter();
          real_type init_rnorm  = solver.getIterativeSolver().getInitResidualNorm();
          real_type final_rnorm = solver.getIterativeSolver().getFinalResidualNorm();
          
          std::cout << "\t IR iterations               : " << num_iter    << " (max " << maxit << ", restart " << restart << ")\n";
          std::cout << "\t IR starting res. norm       : " << init_rnorm  << "\n";
          std::cout << "\t IR final res. norm          : " << final_rnorm << " (tol " << std::setprecision(2) << tol << ")\n\n";
        }

        int checkNormOfScaledResiduals(ReSolve::matrix::Csr& A,
                                      ReSolve::vector::Vector& vec_rhs,
                                      ReSolve::vector::Vector& vec_x,
                                      ReSolve::vector::Vector& vec_r,
                                      ReSolve::SystemSolver& solver)
        {
          using namespace ReSolve::constants;
          using namespace memory;
          int error_sum = 0;

          // Compute norm of scaled residuals for the second system
          real_type inf_norm_A = 0.0;  
          mh_->matrixInfNorm(&A, &inf_norm_A, DEVICE); 
          real_type inf_norm_x = vh_->infNorm(&vec_x, DEVICE);
          real_type inf_norm_r = vh_->infNorm(&vec_r, DEVICE);
          real_type nsr_norm   = inf_norm_r / (inf_norm_A * inf_norm_x);
          real_type nsr_system = solver.getNormOfScaledResiduals(&vec_rhs, &vec_x);
          real_type error      = std::abs(nsr_system - nsr_norm)/nsr_norm;

          if (error > 10.0*std::numeric_limits<real_type>::epsilon()) {
            std::cout << "Norm of scaled residuals computation failed:\n";
            std::cout << std::scientific << std::setprecision(16)
                      << "\tMatrix inf  norm                 : " << inf_norm_A << "\n"
                      << "\tResidual inf norm                : " << inf_norm_r << "\n"  
                      << "\tSolution inf norm                : " << inf_norm_x << "\n"  
                      << "\tNorm of scaled residuals         : " << nsr_norm   << "\n"
                      << "\tNorm of scaled residuals (system): " << nsr_system << "\n\n";
            error_sum++;
          }
          return error_sum;
        }

        int checkRelativeResidualNorm(ReSolve::vector::Vector& vec_rhs,
                                      ReSolve::vector::Vector& vec_x,
                                      const real_type residual_norm,
                                      const real_type rhs_norm,
                                      ReSolve::SystemSolver& solver)
        {
          using namespace memory;
          int error_sum = 0;

          real_type rel_residual_norm = solver.getResidualNorm(&vec_rhs, &vec_x);
          real_type error = std::abs(rhs_norm * rel_residual_norm - residual_norm)/residual_norm;
          if (error > 10.0*std::numeric_limits<real_type>::epsilon()) {
            std::cout << "Relative residual norm computation failed:\n";
            std::cout << std::scientific << std::setprecision(16)
                      << "\tTest value            : " << residual_norm/rhs_norm << "\n"
                      << "\tSystemSolver computed : " << rel_residual_norm   << "\n\n";
            error_sum++;
          }

          return error_sum;
        }

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
