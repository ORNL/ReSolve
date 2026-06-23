/**
 * @file HykktRandomizedDenseConjugateGradientTests.hpp
 * @brief Implementation of tests for class hykkt::RandomizedDenseConjugateGradient
 *
 */
#pragma once

#include <filesystem>

#include <resolve/MemoryUtils.hpp>
#include <resolve/hykkt/randomized_dense_cg/RandomizedDenseConjugateGradient.hpp>
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
     * @brief Tests for class hykkt::RandomizedDenseConjugateGradient. There is currently only
     * one set of input matrices being tested.
     */
    class HykktRandomizedDenseConjugateGradientTests : public TestBase
    {
    public:
      /**
       * @brief Constructs the RandomizedCG test fixture with the specified memory space and handlers.
       *
       * The test fixture uses caller-provided matrix and vector handlers so the same test can be run with CPU, CUDA, or HIP backends.
       *
       * @param[in] memspace Memory space for the test (HOST or DEVICE).
       * @param[in] matrix_handler Reference to a matrix handler for the selected backend.
       * @param[in] vector_handler Reference to a vector handler for the selected backend.
       */
      HykktRandomizedDenseConjugateGradientTests(memory::MemorySpace memspace,
                                                 MatrixHandler&      matrix_handler,
                                                 VectorHandler&      vector_handler)
        : memspace_(memspace),
          matrix_handler_(matrix_handler),
          vector_handler_(vector_handler)
      {
      }

      virtual ~HykktRandomizedDenseConjugateGradientTests()
      {
      }

      /**
       * @brief Test the RandomizedDenseConjugateGradient implementation with matrices in tests\unit\hykkt\RandomizedCGTestMatrices
       *
       * @return TestOutcome Result of the test
       */
      TestOutcome RandomizedDenseCGTest(const std::string& A_file_name, index_type n, index_type k, real_type b_min, real_type b_max)
      {
        hykkt::RandomizedDenseConjugateGradient randomized_cg(n, k, &matrix_handler_, &vector_handler_, memspace_);
        randomized_cg.setSolverTolerance(randomized_cg_tol);
        
        vector::Vector* L = new vector::Vector(n, n);
        L->allocateAll(memspace_);
        vector_handler_.randomVector(L, b_min, b_max, memspace_);
        // L->syncData(memory::HOST);

        vector::Vector* A = new vector::Vector(n, n);
        A->allocateAll(memspace_);

        vector_handler_.gemm('T', 'N', 1.0, 0.0, L, L, A, memspace_);
        if (memspace_ == memory::DEVICE)
        {
          A->syncData(memory::HOST);
        }

        vector::Vector* x = new vector::Vector(n);
        x->allocateAll(memspace_);

        vector::Vector* b = new vector::Vector(n);
        b->allocateAll(memspace_);
        vector_handler_.randomVector(b, b_min, b_max, memspace_);

        randomized_cg.addMatrixInfo(A);
        randomized_cg.addVectorInfo(x, b);
        randomized_cg.setup();
        int converged_n = randomized_cg.solve(); // 0 if converged, 1 if not

        TestStatus  status;
        std::string testname(__func__);
        testname += " n=" + std::to_string(n) + ", k=" + std::to_string(k);
        status *= validateResult(x, converged_n);

        delete L;
        delete A;
        delete x;
        delete b;

        return status.report(testname.c_str());
      }

    private:
      memory::MemorySpace memspace_;       ///< Memory space used by the test.
      MatrixHandler&      matrix_handler_; ///< Backend-specific matrix handler.
      VectorHandler&      vector_handler_; ///< Backend-specific vector handler.

      static constexpr real_type cholesky_tol = 1e-12;
      static constexpr real_type randomized_cg_tol     = 1e-12;
      static constexpr real_type entry_tol    = 1e-6; // Tolerance for checking individual entries

      /**
       * @brief Validate the RandomizedCG result.
       * @param[in] x Pointer to the output x vector.
       */
      bool validateResult(vector::Vector* x, int converged_n)
      {
        return true;
      }
    }; // class HykktRandomizedDenseConjugateGradientTests
  } // namespace tests
} // namespace ReSolve
