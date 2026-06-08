/**
 * @file HykktSolverTests.hpp
 * @author Andrew Xu (xua1@ornl.gov)
 * @brief Implementation of tests for class hykkt::HyKKTSolver
 *
 */
#pragma once

#include <filesystem>

#include <resolve/MemoryUtils.hpp>
#include <resolve/hykkt/HyKKTSolver.hpp>
#include <resolve/matrix/Coo.hpp>
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
     * @brief Tests for class hykkt::HyKKTSolver. This is only
     * a placeholder test. The test will always pass, and a proper validateResult()
     * function needs to be written.
     */
    class HykktSolverTests : public TestBase
    {
    public:
      /**
       * @brief Constructs the solver test fixture with the specified memory space and handlers.
       *
       * The test fixture uses caller-provided matrix and vector handlers so the same test can be run with CPU, CUDA, or HIP backends.
       *
       * @param[in] memspace Memory space for the test (HOST or DEVICE).
       * @param[in] matrix_handler Reference to a matrix handler for the selected backend.
       * @param[in] vector_handler Reference to a vector handler for the selected backend.
       */
      HykktSolverTests(memory::MemorySpace memspace, MatrixHandler& matrix_handler, VectorHandler& vector_handler)
        : memspace_(memspace), matrixHandler_(matrix_handler), vectorHandler_(vector_handler)
      {
      }

      virtual ~HykktSolverTests()
      {
      }

      /**
       * @brief Test the HyKKTSolver implementation with matrices provided by the user or by runHykktSolverTests.cpp
       *
       * @return TestOutcome Result of the test
       */
      TestOutcome
      testSolver(index_type         n_x,
                 index_type         m_d,
                 index_type         m_c,
                 index_type         H_nnz,
                 index_type         D_s_nnz,
                 index_type         J_nnz,
                 index_type         J_d_nnz,
                 const std::string& H_file_name,
                 const std::string& D_s_file_name,
                 const std::string& J_file_name,
                 const std::string& J_d_file_name,
                 const std::string& r_x_file_name,
                 const std::string& r_s_file_name,
                 const std::string& r_y_file_name,
                 const std::string& r_yd_file_name,
                 real_type          gamma)
      {
        constexpr double tol = 1e-2;

        std::ifstream H_file(H_file_name);
        std::ifstream D_s_file(D_s_file_name);
        std::ifstream J_file(J_file_name);
        std::ifstream J_d_file(J_d_file_name);
        std::ifstream r_x_file(r_x_file_name);
        std::ifstream r_s_file(r_s_file_name);
        std::ifstream r_y_file(r_y_file_name);
        std::ifstream r_yd_file(r_yd_file_name);

        // The .mtx file readers write into host accessible memory.
        // Load test data into HOST first, then sync to DEVICE for CUDA and HIP backends.
        matrix::Csr* H   = io::createCsrFromFile(H_file, true);
        matrix::Csr* D_s = io::createCsrFromFile(D_s_file, false);
        matrix::Csr* J   = io::createCsrFromFile(J_file, false);
        matrix::Csr* J_d = io::createCsrFromFile(J_d_file, false);
        if (memspace_ == memory::DEVICE)
        {
          H->allocateMatrixData(memory::DEVICE);
          D_s->allocateMatrixData(memory::DEVICE);
          J->allocateMatrixData(memory::DEVICE);
          J_d->allocateMatrixData(memory::DEVICE);
          H->syncData(memory::DEVICE);
          D_s->syncData(memory::DEVICE);
          J->syncData(memory::DEVICE);
          J_d->syncData(memory::DEVICE);
        }

        // RHS vector blocks
        vector::Vector* r_x  = io::createVectorFromFile(r_x_file);
        vector::Vector* r_s  = io::createVectorFromFile(r_s_file);
        vector::Vector* r_y  = io::createVectorFromFile(r_y_file);
        vector::Vector* r_yd = io::createVectorFromFile(r_yd_file);
        if (memspace_ == memory::DEVICE)
        {
          r_x->allocate(memory::DEVICE);
          r_s->allocate(memory::DEVICE);
          r_y->allocate(memory::DEVICE);
          r_yd->allocate(memory::DEVICE);
          r_x->syncData(memory::DEVICE);
          r_s->syncData(memory::DEVICE);
          r_y->syncData(memory::DEVICE);
          r_yd->syncData(memory::DEVICE);
        }

        // LHS vector blocks
        vector::Vector* x   = new vector::Vector(n_x);
        vector::Vector* s   = new vector::Vector(m_d);
        vector::Vector* y   = new vector::Vector(m_c);
        vector::Vector* y_d = new vector::Vector(m_d);
        x->allocateAll(memspace_);
        s->allocateAll(memspace_);
        y->allocateAll(memspace_);
        y_d->allocateAll(memspace_);

        hykkt::HyKKTSolver hykktSolver(n_x, m_d, m_c, memspace_);
        hykktSolver.setMatrixBlocks(H, D_s, J, J_d);
        hykktSolver.setRHSBlocks(r_x, r_s, r_y, r_yd);
        hykktSolver.setLHSPointers(x, s, y, y_d);
        hykktSolver.setGamma(gamma);
        hykktSolver.addHandlers(&matrixHandler_, &vectorHandler_);

        real_type error = hykktSolver.solve();

        TestStatus  status;
        std::string testname(__func__);
        index_type  N   = n_x + m_c + 2 * m_d;
        index_type  nnz = H_nnz + D_s_nnz + J_nnz + J_d_nnz;
        testname += " N=" + std::to_string(N) + ", nnz =" + std::to_string(nnz) + '\n';
        status *= validateResult(error, tol);

        delete H;
        delete D_s;
        delete J;
        delete J_d;
        delete r_x;
        delete r_s;
        delete r_y;
        delete r_yd;
        delete x;
        delete s;
        delete y;
        delete y_d;

        return status.report(testname.c_str());
      }

    private:
      memory::MemorySpace memspace_;      ///< Memory space used by the test.
      MatrixHandler&      matrixHandler_; ///< Backend-specific matrix handler.
      VectorHandler&      vectorHandler_; ///< Backend-specific vector handler.

      /**
       * @brief Validate the solver result.
       * @param[in] error Error of Ax - b, where x is the result obtained by the solver.
       * @param[in] tol Solver tolerance.
       */
      bool validateResult(real_type error, real_type tol)
      {
        return error < tol;
      }
    }; // class HykktSolverTests
  } // namespace tests
} // namespace ReSolve
