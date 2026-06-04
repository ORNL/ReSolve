/**
 * @file HykktSchurComplementConjugateGradientTests.hpp
 * @brief Implementation of tests for class hykkt::SchurComplementConjugateGradient
 *
 */
#pragma once

#include <filesystem>

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
     * @brief Tests for class hykkt::SchurComplementConjugateGradient. There is currently only
     * one set of input matrices being tested.
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
       */
      HykktSchurComplementConjugateGradientTests(memory::MemorySpace memspace,
                                                 MatrixHandler&      matrix_handler,
                                                 VectorHandler&      vector_handler)
        : memspace_(memspace),
          matrix_handler_(matrix_handler),
          vector_handler_(vector_handler)
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
        std::ifstream jcFile(jcFileName);
        std::ifstream hFile(hFileName);
        std::ifstream bFile(bFileName);

        matrix::Csr* h  = io::createCsrFromFile(hFile, true);
        matrix::Csr* jc = io::createCsrFromFile(jcFile, false);
        if (memspace_ == memory::DEVICE)
        {
          h->syncData(memory::DEVICE);
          jc->syncData(memory::DEVICE);
        }
        hykkt::CholeskySolver choleskySolver(memspace_);
        choleskySolver.addMatrixInfo(h);
        choleskySolver.symbolicAnalysis();
        choleskySolver.setPivotTolerance(cholesky_tol);
        choleskySolver.numericalFactorization();

        index_type                              n   = jc->getNumRows();
        index_type                              m   = jc->getNumColumns();
        index_type                              nnz = jc->getNnz();
        hykkt::SchurComplementConjugateGradient sccg(n, m, &choleskySolver, &matrix_handler_, &vector_handler_, memspace_);
        sccg.setSolverTolerance(sccg_tol);

        matrix::Csr* jc_tr = new matrix::Csr(m, n, nnz);
        jc_tr->allocateMatrixData(memspace_);
        matrix_handler_.transpose(jc, jc_tr, memspace_);

        vector::Vector* x0 = new vector::Vector(n);
        x0->allocate(memspace_);

        vector::Vector* b = io::createVectorFromFile(bFile);
        if (memspace_ == memory::DEVICE)
        {
          b->syncData(memory::DEVICE);
        }

        sccg.addMatrixInfo(jc, jc_tr);
        sccg.addVectorInfo(x0, b);
        sccg.setup();
        int converged_n = sccg.solve(); // 0 if converged, 1 if not

        TestStatus  status;
        std::string testname(__func__);
        testname += " n=" + std::to_string(n) + ", m=" + std::to_string(m) + ", nnz =" + std::to_string(nnz);
        status *= validateResult(x0, converged_n);

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

      static constexpr real_type cholesky_tol = 1e-12;
      static constexpr real_type sccg_tol     = 1e-12;
      static constexpr real_type entry_tol    = 1e-6; // Tolerance for checking individual entries

      // Expected outputs. Currently they are only available for the set of matrix files below
      static constexpr real_type x_0_expected = 22.171865776354700;
      static constexpr real_type x_6_expected = -4.446628667344612e+03;

      std::string source_dir = std::string(SOURCE_DIR);
      std::string jcFileName = source_dir + "/SCCGTestMatrices/JC_matrix_ACTIVSg200_AC_00.mtx";
      std::string hFileName  = source_dir + "/SCCGTestMatrices/H_matrix_ACTIVSg200_AC_00.mtx";
      std::string bFileName  = source_dir + "/SCCGTestMatrices/CG_rhs_ACTIVSg200_AC_00.mtx"; // rhs

      static constexpr real_type cholesky_tol = 1e-12;
      static constexpr real_type sccg_tol = 1e-12;
      static constexpr real_type entry_tol = 1e-6; // Tolerance for checking individual entries

      // Expected outputs. Currently they are only available for the set of matrix files below
      static constexpr real_type x_0_expected = 22.171865776354700;
      static constexpr real_type x_6_expected = -4.446628667344612e+03;

      std::string   source_dir   = std::string(SOURCE_DIR);
      std::string   jcFileName = source_dir + "/SCCGTestMatrices/JC_matrix_ACTIVSg200_AC_00.mtx";
      std::string   hFileName  = source_dir + "/SCCGTestMatrices/H_matrix_ACTIVSg200_AC_00.mtx";
      std::string   bFileName  = source_dir + "/SCCGTestMatrices/CG_rhs_ACTIVSg200_AC_00.mtx"; // rhs

      /**
       * @brief Validate the SCCG result.
       * @param[in] x0 Pointer to the output x0 vector.
       */
      bool validateResult(vector::Vector* x0, int converged_n)
      {
        if (converged_n != 0)
        {
          return false;
        }
        if (memspace_ == memory::DEVICE)
        {
          x0->syncData(memory::HOST);
        }
        if (std::abs(x0->getData(memory::HOST)[0] - x_0_expected) > entry_tol)
        {
          return false;
        }
        if (std::abs(x0->getData(memory::HOST)[6] - x_6_expected) > entry_tol)
        {
          return false;
        }
        return true;
      }
    }; // class HykktSchurComplementConjugateGradientTests
  } // namespace tests
} // namespace ReSolve
