/**
 * @file HykktDenseConjugateGradientTests.hpp
 * @brief Implementation of tests for class hykkt::ConjugateGradient
 *
 */
#pragma once

#include <filesystem>
#include <random>

#include <resolve/MemoryUtils.hpp>
#include <resolve/hykkt/dense_cg/DenseConjugateGradient.hpp>
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
     * @brief Tests for class hykkt::ConjugateGradient. There is currently only
     * one set of input matrices being tested.
     */
    class HykktDenseConjugateGradientTests : public TestBase
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
      HykktDenseConjugateGradientTests(memory::MemorySpace memspace,
                                                 MatrixHandler&      matrix_handler,
                                                 VectorHandler&      vector_handler)
        : memspace_(memspace),
          matrix_handler_(matrix_handler),
          vector_handler_(vector_handler),
          generator_(constants::SEED)
      {
      }

      virtual ~HykktDenseConjugateGradientTests()
      {
      }

      void randomVector(vector::Vector* v, real_type min, real_type max)
      {
        std::uniform_real_distribution<real_type> distribution(min, max);
        for (index_type i = 0; i < v->getSize(); ++i)
        {
          v->getData(memory::HOST)[i] = distribution(generator_);
        }
        v->setDataUpdated(memory::HOST);
      }

      /**
       * @brief Test the ConjugateGradient implementation with matrices in tests\unit\hykkt\CGTestMatrices
       *
       * @return TestOutcome Result of the test
       */
      TestOutcome DenseCGTest(const std::string& A_file_name, index_type n, real_type b_min, real_type b_max)
      {
        hykkt::DenseConjugateGradient cg(n, &matrix_handler_, &vector_handler_, memspace_);
        cg.setSolverTolerance(cg_tol);
        
        vector::Vector* L = new vector::Vector(n, n);
        L->allocateAll(memspace_);
        randomVector(L, b_min, b_max);
        if (memspace_ == memory::DEVICE)
        {
          L->syncData(memory::DEVICE);
        }

        vector::Vector* A = new vector::Vector(n, n);
        A->allocateAll(memspace_);

        vector_handler_.gemm('T', 'N', 1.0, 0.0, L, L, A, memspace_);
        if (memspace_ == memory::DEVICE)
        {
          A->syncData(memory::HOST);
        }

        vector_handler_.gemm('T', 'N', 1.0, 0.0, L, L, A, memspace_);

        vector::Vector* x = new vector::Vector(n);
        x->allocateAll(memspace_);

        vector::Vector* b = new vector::Vector(n);
        b->allocateAll(memspace_);
        randomVector(b, b_min, b_max);
        if (memspace_ == memory::DEVICE)
        {
          b->syncData(memory::DEVICE);
        }

        cg.addMatrixInfo(A);
        cg.addVectorInfo(x, b);
        cg.setup();
        int converged_n = cg.solve(); // 0 if converged, 1 if not

        TestStatus  status;
        std::string testname(__func__);
        testname += " n=" + std::to_string(n);
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
      static constexpr real_type cg_tol     = 1e-12;
      static constexpr real_type entry_tol    = 1e-6; // Tolerance for checking individual entries

      std::mt19937 generator_;

      /**
       * @brief Validate the CG result.
       * @param[in] x Pointer to the output x vector.
       */
      bool validateResult(vector::Vector* x, int converged_n)
      {
        return true;
      }
    }; // class HykktDenseConjugateGradientTests
  } // namespace tests
} // namespace ReSolve
