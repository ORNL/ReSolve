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
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/matrix/io.hpp>
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
      HykktSchurComplementConjugateGradientTests(memory::MemorySpace memspace, MatrixHandler& matrixHandler, VectorHandler& vectorHandler)
        : memspace_(memspace), matrixHandler_(matrixHandler), vectorHandler_(vectorHandler)
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

        std::string   sourceDir  = std::string(SOURCE_DIR);
        std::string   jcFileName = sourceDir + "/SCCGTestMatrices/JC_matrix_ACTIVSg200_AC_00.mtx";
        std::string   hFileName  = sourceDir + "/SCCGTestMatrices/H_matrix_ACTIVSg200_AC_00.mtx";
        std::string   bFileName  = sourceDir + "/SCCGTestMatrices/CG_rhs_ACTIVSg200_AC_00.mtx"; // rhs
        std::ifstream jcFile(jcFileName);
        std::ifstream hFile(hFileName);
        std::ifstream bFile(bFileName);

        matrix::Csr* h = new matrix::Csr(2278, 2278, 11304, true, false);
        h->allocateMatrixData(memspace_);
        io::updateMatrixFromFile(hFile, h);
        hykkt::CholeskySolver choleskySolver(memspace_);
        choleskySolver.addMatrixInfo(h);
        choleskySolver.symbolicAnalysis();
        choleskySolver.setPivotTolerance(tol);
        choleskySolver.numericalFactorization();

        matrix::Csr* jc = new matrix::Csr(1386, 2278, 6784, false, false);
        jc->allocateMatrixData(memspace_);
        io::updateMatrixFromFile(jcFile, jc);

        index_type                              n   = jc->getNumRows();
        index_type                              m   = jc->getNumColumns();
        index_type                              nnz = jc->getNnz();
        hykkt::SchurComplementConjugateGradient sccg(n, m, &choleskySolver, memspace_, matrixHandler_, vectorHandler_);
        sccg.setSolverTolerance(tol);

        matrix::Csr* jc_tr = new matrix::Csr(m, n, nnz);
        jc_tr->allocateMatrixData(memspace_);
        matrixHandler_.transpose(jc, jc_tr, memspace_);

        vector::Vector* x0 = new vector::Vector(n);
        x0->allocate(memspace_);
        randomVector(x0);

        vector::Vector* b = new vector::Vector(n);
        b->allocate(memspace_);
        io::updateVectorFromFile(bFile, b);

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
      memory::MemorySpace memspace_;
      MatrixHandler&      matrixHandler_;
      VectorHandler&      vectorHandler_;

      /**
       * @brief Generate a random vector of doubles between 0 and RAND_MAX. Copied from HykktCholeskyTests.hpp.
       * @param[in] vec Target vector to write to.
       */
      void randomVector(vector::Vector* vec)
      {
        for (index_type i = 0; i < vec->getSize(); ++i)
        {
          vec->getData(memory::HOST)[i] = static_cast<double>(rand()) / RAND_MAX;
        }
        vec->setDataUpdated(memory::HOST);
        if (memspace_ == memory::DEVICE)
        {
          vec->syncData(memory::DEVICE);
        }
      }

      /**
       * @brief Validate the results of the scaling
       * @param[in] x0 Pointer to the output x0 vector
       * @param[in] tol Solver tolerance
       */
      bool validateResult(vector::Vector* x0, real_type tol)
      {
        bool test_passed = true;

        // To be implemented

        return test_passed;
      }
    }; // class HykktPermutationTests
  } // namespace tests
} // namespace ReSolve
