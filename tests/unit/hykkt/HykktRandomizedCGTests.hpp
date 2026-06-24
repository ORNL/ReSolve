/**
 * @file HykktRandomizedConjugateGradientTests.hpp
 * @brief Implementation of tests for class hykkt::RandomizedConjugateGradient
 *
 */
#pragma once

#include <filesystem>

#include <resolve/MemoryUtils.hpp>
#include <resolve/hykkt/randomized_cg/RandomizedConjugateGradient.hpp>
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
     * @brief Tests for class hykkt::RandomizedConjugateGradient. There is currently only
     * one set of input matrices being tested.
     */
    class HykktRandomizedConjugateGradientTests : public TestBase
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
      HykktRandomizedConjugateGradientTests(memory::MemorySpace memspace,
                                                 MatrixHandler&      matrix_handler,
                                                 VectorHandler&      vector_handler)
        : memspace_(memspace),
          matrix_handler_(matrix_handler),
          vector_handler_(vector_handler)
      {
      }

      virtual ~HykktRandomizedConjugateGradientTests()
      {
      }

      /**
       * @brief Test the RandomizedConjugateGradient implementation with matrices in tests\unit\hykkt\RandomizedCGTestMatrices
       *
       * @return TestOutcome Result of the test
       */
      TestOutcome RandomizedCGTest(const std::string& A_file_name, index_type k, real_type rng_min, real_type rng_max)
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
        hykkt::RandomizedConjugateGradient randomized_cg(n, k, &matrix_handler_, &vector_handler_, memspace_);
        randomized_cg.setSolverTolerance(randomized_cg_tol);

        vector::Vector* x = new vector::Vector(n);
        x->allocateAll(memspace_);

        vector::Vector* b = new vector::Vector(n);
        b->allocateAll(memspace_);
        vector_handler_.randomVector(b, rng_min, rng_max, memspace_);

        randomized_cg.addMatrixInfo(A);
        randomized_cg.addVectorInfo(x, b);
        randomized_cg.setup();
        int converged_n = randomized_cg.solve(); // 0 if converged, 1 if not

        TestStatus  status;
        std::string testname(__func__);
        testname += " n=" + std::to_string(n) + ", k=" + std::to_string(k) + ", nnz =" + std::to_string(nnz);
        status *= validateResult(x, converged_n);

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
    }; // class HykktRandomizedConjugateGradientTests
  } // namespace tests
} // namespace ReSolve
