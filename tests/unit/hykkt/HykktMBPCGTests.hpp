/**
 * @file HykktMultiBasisParallelConjugateGradientTests.hpp
 * @brief Implementation of tests for class hykkt::MultiBasisParallelConjugateGradient
 *
 */
#pragma once

#include <filesystem>

#include <resolve/MemoryUtils.hpp>
#include <resolve/hykkt/mbpcg/MultiBasisParallelConjugateGradient.hpp>
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
     * @brief Tests for class hykkt::MultiBasisParallelConjugateGradient. There is currently only
     * one set of input matrices being tested.
     */
    template <typename WorkspaceType>
    class HykktMultiBasisParallelConjugateGradientTests : public TestBase
    {
    public:
      /**
       * @brief Constructs the MBPCG test fixture with the specified memory space and handlers.
       *
       * The test fixture uses caller-provided matrix and vector handlers so the same test can be run with CPU, CUDA, or HIP backends.
       *
       * @param[in] memspace Memory space for the test (HOST or DEVICE).
       * @param[in] matrix_handler Reference to a matrix handler for the selected backend.
       * @param[in] vector_handler Reference to a vector handler for the selected backend.
       */
      HykktMultiBasisParallelConjugateGradientTests(memory::MemorySpace memspace,
                                                 MatrixHandler&      matrix_handler,
                                                 VectorHandler&      vector_handler,
                                                 WorkspaceType&      workspace)
        : memspace_(memspace),
          matrix_handler_(matrix_handler),
          vector_handler_(vector_handler),
          workspace_(workspace)
      {
      }

      virtual ~HykktMultiBasisParallelConjugateGradientTests()
      {
      }

      /**
       * @brief Test the MultiBasisParallelConjugateGradient implementation with matrices in tests\unit\hykkt\MBPCGTestMatrices
       *
       * @return TestOutcome Result of the test
       */
      TestOutcome MBPCGTest(const std::string& A_file_name, const std::string& b_file_name = "", bool use_file_for_b = false)
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
        int num_fails = 0;

        printf("\nTesting with diagonal scaling.\n");
        // k = 1, 2, 4, 8
        for (index_type k = 1; k <= 8; k *= 2)
        {
          if (k == 8) continue;
          printf("\nk=%d\n", k);
          hykkt::MultiBasisParallelConjugateGradient mbpcg(n, k, &matrix_handler_, &vector_handler_, memspace_);
          mbpcg.setSolverTolerance(initial_tol, convergence_tol);
          mbpcg.setSolverItmax(2000);

          mbpcg.addMatrixInfo(A);
          mbpcg.addVectorInfo(x, b);
          mbpcg.diagonalScale();
        
          mbpcg.setup();
          int converged = mbpcg.solve(); // 0 if converged, 1 if not
          num_fails += (converged != 0);
        }

        printf("\nTesting with preconditioner.\n");
        PreconditionerIChol0 preconditioner(&matrix_handler_, &workspace_);
        if (!preconditioner.setup(A))
        {
          bool found = false;
          real_type inf_norm;
          matrix_handler_.matrixInfNorm(A, &inf_norm, memspace_);
          real_type numeric_boost = inf_norm / 32.0;
          for (index_type i = 0; i < 10; i++)
          {
            printf("Trying boost value %f\n", numeric_boost);
            preconditioner.setNumericBoost(numeric_boost);
            if (!preconditioner.setup(A)) // ANDREW TODO: this doesn't work on hip because rocblas doesn't report singularities (or maybe it doesn but just very rarely)
            {
              found = true;
              break;
            }
            numeric_boost *= 2.0;
          }
          if (!found)
          {
            out::error() << "Preconditioning failed!";
          }
        }

        // k = 1, 2, 4, 8
        for (index_type k = 1; k <= 8; k *= 2)
        {
          // if (k != 8) continue;
          printf("\nk=%d\n", k);
          preconditioner.setNumRhs(k);
          hykkt::MultiBasisParallelConjugateGradient mbpcg(n, k, &preconditioner, &matrix_handler_, &vector_handler_, memspace_);
          mbpcg.setSolverTolerance(initial_tol, convergence_tol);
          mbpcg.setSolverItmax(2000);

          mbpcg.addMatrixInfo(A);
          mbpcg.addVectorInfo(x, b);
          mbpcg.setup();
          int converged = mbpcg.solve(); // 0 if converged, 1 if not
          num_fails += (converged != 0);
        }

        std::string testname(__func__);
        testname += " n=" + std::to_string(n) + ", nnz=" + std::to_string(nnz);
        status *= (num_fails == 0);

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

      static constexpr real_type initial_tol = 1e-8;
      static constexpr real_type convergence_tol     = 1e-8;
    }; // class HykktMultiBasisParallelConjugateGradientTests
  } // namespace tests
} // namespace ReSolve
