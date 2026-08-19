/**
 * @file HykktConjugateGradientTests.hpp
 * @brief Implementation of tests for class hykkt::ConjugateGradient
 *
 */
#pragma once

#include <filesystem>

#include <resolve/MemoryUtils.hpp>
#include <resolve/hykkt/cg/ConjugateGradient.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/preconditioner_ichol0/PreconditionerIChol0.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <tests/unit/TestBase.hpp>

namespace ReSolve
{
  namespace tests
  {
    /**
     * @brief Tests for class hykkt::ConjugateGradient. There is currently only
     * one set of input matrices being tested.
     */
    template <typename WorkspaceType>
    class HykktConjugateGradientTests : public TestBase
    {
    public:
      /**
       * @brief Constructs the CG test fixture with the specified memory space and handlers.
       *
       * The test fixture uses caller-provided matrix and vector handlers so the same test can be run with CPU, CUDA, or HIP backends.
       *
       * @param[in] memspace Memory space for the test (HOST or DEVICE).
       * @param[in] matrix_handler Reference to a matrix handler for the selected backend.
       * @param[in] vector_handler Reference to a vector handler for the selected backend.
       */
      HykktConjugateGradientTests(memory::MemorySpace memspace,
                                                 MatrixHandler&      matrix_handler,
                                                 VectorHandler&      vector_handler,
                                                 WorkspaceType&      workspace)
        : memspace_(memspace),
          matrix_handler_(matrix_handler),
          vector_handler_(vector_handler),
          workspace_(workspace)
      {
      }

      virtual ~HykktConjugateGradientTests()
      {
      }

      /**
       * @brief Test the ConjugateGradient implementation with matrices in tests\unit\hykkt\CGTestMatrices
       * 
       * @param[in] A_file_name Path of the .mtx file for the matrix A
       * @param[in] b_file_name Path of the .mtx file for the RHS vector b. Optional
       * @param[in] use_file_for_b Choose whether to load a file from b_file_name to use for the RHS. If false, randomly generate b
       *
       * @return TestOutcome Result of the test
       */
      TestOutcome CGTest(const std::string& A_file_name, const std::string& b_file_name = "", bool use_file_for_b = false)
      {
        std::ifstream A_file(A_file_name);

        matrix::Csr* A = io::createCsrFromFile(A_file, true);
        if (memspace_ == memory::DEVICE)
        {
          A->allocateMatrixData(memory::DEVICE);
          A->syncData(memory::DEVICE);
        }

        index_type                              n   = A->getNumRows();
        index_type                              nnz = A->getNnz();

        vector::Vector* b;
        if (use_file_for_b)
        {
          std::ifstream b_file(b_file_name);
          b = io::createVectorFromFile(b_file);
          if (memspace_ == memory::DEVICE)
          {
            b->syncData(memory::DEVICE);
          }
        }
        else
        {
          b = new vector::Vector(n);
          b->allocateAll(memspace_);
          vector_handler_.randomVector(b, -1.0, 1.0, memspace_);
        }

        vector::Vector* x = new vector::Vector(n);
        x->allocate(memspace_);

        TestStatus  status;
      
        printf("\nTesting with diagonal scaling.\n");
        hykkt::ConjugateGradient cg_diag_scal(n, &matrix_handler_, &vector_handler_, memspace_);
        cg_diag_scal.setSolverTolerance(cg_tol_);
        // cg_diag_scal.setSolverItmax();

        x->setToZero(memspace_);
        cg_diag_scal.addMatrixInfo(A);
        cg_diag_scal.addVectorInfo(x, b);
        cg_diag_scal.diagonalScale();
        cg_diag_scal.setup();
        int converged = cg_diag_scal.solve(); // 0 if converged, 1 if not
        status *= (converged == 0);
        
#ifdef RESOLVE_USE_GPU
        printf("\nTesting with preconditioner.\n");
        PreconditionerIChol0 preconditioner(&matrix_handler_, &workspace_);
        preconditioner.setNumericBoost(numeric_boost_);
        preconditioner.setup(A);

        hykkt::ConjugateGradient cg_prec(n, &preconditioner, &matrix_handler_, &vector_handler_, memspace_);
        cg_prec.setSolverTolerance(cg_tol_);
        // cg_prec.setSolverItmax();

        x->setToZero(memspace_);
        cg_prec.addMatrixInfo(A);
        cg_prec.addVectorInfo(x, b);
        cg_prec.setup();
        converged = cg_prec.solve(); // 0 if converged, 1 if not
        status *= (converged == 0);
#endif
      
        std::string testname(__func__);
        testname += " n=" + std::to_string(n) + ", nnz =" + std::to_string(nnz);

        delete A;
        delete x;
        delete b;

        return status.report(testname.c_str());
      }

    private:
      memory::MemorySpace memspace_;       ///< Memory space used by the test.
      MatrixHandler&      matrix_handler_; ///< Backend-specific matrix handler.
      VectorHandler&      vector_handler_; ///< Backend-specific vector handler.
      WorkspaceType&      workspace_;

      static constexpr real_type cholesky_tol_ = 1e-12;
      static constexpr real_type cg_tol_     = 1e-8;
      static constexpr real_type numeric_boost_ = 437621.0;
    }; // class HykktConjugateGradientTests
  } // namespace tests
} // namespace ReSolve
