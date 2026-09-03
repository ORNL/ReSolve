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
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <tests/unit/TestBase.hpp>

namespace ReSolve
{
  using out = io::Logger;
  
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
      TestOutcome CGTest(const std::string& A_file_name, const std::string& b_file_name = "", bool use_file_for_b = false) // ANDREW TODO: get rid of bool argument
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
        int converged;
      
        // printf("\nTesting with diagonal scaling.\n");
        // hykkt::ConjugateGradient cg_diag_scal(n, &matrix_handler_, &vector_handler_, memspace_);
        // cg_diag_scal.setSolverTolerance(cg_tol_);
        // cg_diag_scal.setSolverItmax(12000);

        // x->setToZero(memspace_);
        // cg_diag_scal.addMatrixInfo(A);
        // cg_diag_scal.addVectorInfo(x, b);
        // cg_diag_scal.diagonalScale();
        // cg_diag_scal.setup();
        // int converged = cg_diag_scal.solve(); // 0 if converged, 1 if not
        // status *= (converged == 0);
        
#ifdef RESOLVE_USE_GPU
        printf("\nTesting with preconditioner.\n");
        PreconditionerIChol0 preconditioner(&matrix_handler_, &workspace_);
        
        real_type numeric_boost = 0.0;
        bool found_valid_boost = (preconditioner.setup(A) == 0);
        if (!found_valid_boost)
        {
          real_type inf_norm;
          matrix_handler_.matrixInfNorm(A, &inf_norm, memspace_);
          numeric_boost = inf_norm / 4194304.0;
          for (index_type i = 0; i < 14; i++)
          {
            printf("Trying boost value %f\n", numeric_boost);
            preconditioner.setNumericBoost(numeric_boost);
            if (preconditioner.setup(A) == 0)
            {
              found_valid_boost = true;
              break;
            }
            numeric_boost *= 2.0;
          }
        }
        
        if (!found_valid_boost)
        {
          printf("No valid numerical boost value found!\n");
          status *= 0;
        }
        else
        {
          printf("Using boost value %f\n", numeric_boost);
          hykkt::ConjugateGradient cg_prec(n, &preconditioner, &matrix_handler_, &vector_handler_, memspace_);
          cg_prec.setSolverTolerance(cg_tol_);
          cg_prec.setSolverItmax(12000);

          x->setToZero(memspace_);
          cg_prec.addMatrixInfo(A);
          cg_prec.addVectorInfo(x, b);
          cg_prec.setup();
          converged = cg_prec.solve(); // 0 if converged, 1 if not
          status *= (converged == 0);
        }
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
    }; // class HykktConjugateGradientTests
  } // namespace tests
} // namespace ReSolve
