/**
 * @file HykktSchurComplementConjugateGradientTests.hpp
 * @brief Implementation of tests for class hykkt::SchurComplementConjugateGradient
 *
 */
#pragma once

#include <filesystem>
#include <random>

#include <resolve/MemoryUtils.hpp>
#include <resolve/hykkt/sccg/SchurComplementConjugateGradient.hpp>
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
     * @brief Tests for class hykkt::SchurComplementConjugateGradient. This is only
     * a placeholder test. The test will always pass, and a proper validateResult()
     * function needs to be written.
     */
    class HykktSchurComplementConjugateGradientTests : public TestBase
    {
    public:
      /**
       * @brief Constructs the SCCG test fixture with the specified memory space and handlers.
       *
       * The test fixture uses caller-provided matrix and vector handlers so the same test can be run with CPU, CUDA, or HIP backends.
       *
       * @param[in] memspace Memory space for the test (HOST or DEVICE).
       * @param[in] matrix_handler Reference to a matrix handler for the selected backend.
       * @param[in] vector_handler Reference to a vector handler for the selected backend.
       * @param[in] generator Reference to a C++ random number generator.
       */
      HykktSchurComplementConjugateGradientTests(memory::MemorySpace memspace,
                                                 MatrixHandler&      matrix_handler,
                                                 VectorHandler&      vector_handler,
                                                 std::mt19937&       generator)
        : memspace_(memspace),
          matrix_handler_(matrix_handler),
          vector_handler_(vector_handler),
          generator_(generator)
      {
      }

      virtual ~HykktSchurComplementConjugateGradientTests()
      {
      }

      /**
       * @brief Test the SchurComplementConjugateGradient implementation with matrices in tests\unit\hykkt\SCCGTestMatrices
       *
       * @return TestOutcome Result of the test
       */
      TestOutcome SCCGTest()
      {
        constexpr double tol = 1e-12;

        std::string   source_dir   = std::string(SOURCE_DIR);
        std::string   jc_file_name = source_dir + "/SCCGTestMatrices/JC_matrix_ACTIVSg200_AC_00.mtx";
        std::string   h_file_name  = source_dir + "/SCCGTestMatrices/H_matrix_ACTIVSg200_AC_00.mtx";
        std::string   b_file_name  = source_dir + "/SCCGTestMatrices/CG_rhs_ACTIVSg200_AC_00.mtx"; // rhs
        std::ifstream jc_file(jc_file_name);
        std::ifstream h_file(h_file_name);
        std::ifstream b_file(b_file_name);

        // The .mtx file readers write into host accessible memory.
        // Load test data into HOST first, then sync to DEVICE for CUDA and HIP backends.
        matrix::Csr* h = new matrix::Csr(2278, 2278, 11304, true, false);
        h->allocateMatrixData(memory::HOST);
        io::updateMatrixFromFile(h_file, h);
        if (memspace_ == memory::DEVICE)
        {
          h->syncData(memory::DEVICE);
        }
        hykkt::CholeskySolver choleskySolver(memspace_);
        choleskySolver.addMatrixInfo(h);
        choleskySolver.symbolicAnalysis();
        choleskySolver.setPivotTolerance(tol);
        choleskySolver.numericalFactorization();

        matrix::Csr* jc = new matrix::Csr(1386, 2278, 6784, false, false);
        jc->allocateMatrixData(memory::HOST);
        io::updateMatrixFromFile(jc_file, jc);
        if (memspace_ == memory::DEVICE)
        {
          jc->syncData(memory::DEVICE);
        }

        index_type                              n   = jc->getNumRows();
        index_type                              m   = jc->getNumColumns();
        index_type                              nnz = jc->getNnz();
        hykkt::SchurComplementConjugateGradient sccg(n, m, &choleskySolver, memspace_, matrix_handler_, vector_handler_);
        sccg.setSolverTolerance(tol);

        matrix::Csr* jc_tr = new matrix::Csr(m, n, nnz);
        jc_tr->allocateMatrixData(memspace_);
        matrix_handler_.transpose(jc, jc_tr, memspace_);

        vector::Vector* x0 = new vector::Vector(n);
        x0->allocate(memory::HOST);
        randomVector(x0);

        vector::Vector* b = new vector::Vector(n);
        b->allocate(memory::HOST);
        io::updateVectorFromFile(b_file, b);
        if (memspace_ == memory::DEVICE)
        {
          b->syncData(memory::DEVICE);
        }

        sccg.addMatrixInfo(jc, jc_tr);
        sccg.addVectorInfo(x0, b);
        sccg.setup();
        sccg.solve();

        TestStatus  status;
        std::string testname(__func__);
        testname += " n=" + std::to_string(n) + ", m=" + std::to_string(m) + ", nnz =" + std::to_string(nnz);
        status *= validateResult(x0, tol);

        delete h;
        delete jc;
        delete jc_tr;
        delete x0;
        delete b;

        return status.report(testname.c_str());
      }

    private:
      memory::MemorySpace memspace_;       ///< Memory space used by the test.
      MatrixHandler&      matrix_handler_; ///< Backend-specific matrix handler.
      VectorHandler&      vector_handler_; ///< Backend-specific vector handler.
      std::mt19937&       generator_;      ///< C++ random number generator.

      /**
       * @brief Generate a random vector of doubles between 0 and 1. Copied from HykktCholeskyTests.hpp.
       * @param[in] vec Target vector to write to.
       */
      void randomVector(vector::Vector* vec)
      {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        for (index_type i = 0; i < vec->getSize(); ++i)
        {
          vec->getData(memory::HOST)[i] = distribution(generator_);
        }
        vec->setDataUpdated(memory::HOST);
        if (memspace_ == memory::DEVICE)
        {
          vec->syncData(memory::DEVICE);
        }
      }

      /**
       * @brief Validate the SCCG result.
       * @param[in] x0 Pointer to the output x0 vector.
       * @param[in] tol Solver tolerance.
       */
      bool validateResult(vector::Vector*, real_type)
      {
        bool test_passed = true;

        // To be implemented

        return test_passed;
      }
    }; // class HykktSchurComplementConjugateGradientTests
  } // namespace tests
} // namespace ReSolve
