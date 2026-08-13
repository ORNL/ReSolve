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
#include <resolve/vector/VectorHandler.hpp>
#include <tests/unit/TestBase.hpp>

namespace ReSolve
{
  namespace tests
  {
    /**
     * @brief Tests for class hykkt::MultiBasisParallelConjugateGradient. There is currently only
     * one set of input matrices being tested.
     */
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
                                                 VectorHandler&      vector_handler)
        : memspace_(memspace),
          matrix_handler_(matrix_handler),
          vector_handler_(vector_handler)
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
      TestOutcome MBPCGTest(const std::string& A_file_name, real_type rng_min, real_type rng_max)
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

        vector::Vector* x = new vector::Vector(n);
        x->allocate(memspace_);

        vector::Vector* b = new vector::Vector(n);
        b->allocate(memspace_);
        vector_handler_.randomVector(b, rng_min, rng_max, memspace_);

        int num_converged = 0;
        // k = 1, 2, 4, 8
        for (index_type k = 1; k <= 8; k *= 2)
        {
          // if (k == 1) continue;
          printf("\nk=%d\n", k);
          hykkt::MultiBasisParallelConjugateGradient mbpcg(n, k, &matrix_handler_, &vector_handler_, memspace_);
          mbpcg.setSolverTolerance(initial_tol, convergence_tol);
          // mbpcg.setSolverItmax();

          mbpcg.addMatrixInfo(A);
          mbpcg.addVectorInfo(x, b);
          mbpcg.diagonalScale();
          mbpcg.setup();
          int converged = mbpcg.solve(); // 0 if converged, 1 if not
          num_converged += (converged == 0);
        }
        
        TestStatus  status;
        std::string testname(__func__);
        testname += " n=" + std::to_string(n) + ", nnz=" + std::to_string(nnz);
        status *= (num_converged == 4);

        delete A;
        delete x;
        delete b;

        return status.report(testname.c_str());
      }

    private:
      memory::MemorySpace memspace_;       ///< Memory space used by the test.
      MatrixHandler&      matrix_handler_; ///< Backend-specific matrix handler.
      VectorHandler&      vector_handler_; ///< Backend-specific vector handler.

      static constexpr real_type initial_tol = 1e-8;
      static constexpr real_type convergence_tol     = 1e-8;
      static constexpr real_type entry_tol    = 1e-6; // Tolerance for checking individual entries
    }; // class HykktMultiBasisParallelConjugateGradientTests
  } // namespace tests
} // namespace ReSolve
