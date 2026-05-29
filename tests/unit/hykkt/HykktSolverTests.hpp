/**
 * @file HykktSolverTests.hpp
 * @author Andrew Xu (xua1@ornl.gov)
 * @brief Implementation of tests for class hykkt::HyKKTSolver
 *
 */
#pragma once

#include <filesystem>
#include <random>

#include <resolve/MemoryUtils.hpp>
#include <resolve/hykkt/HyKKTSolver.hpp>
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
       * @param[in] generator Reference to a C++ random number generator.
       */
      HykktSolverTests(memory::MemorySpace memspace, MatrixHandler& matrix_handler, VectorHandler& vector_handler, std::mt19937& generator)
        : memspace_(memspace), matrixHandler_(matrix_handler), vectorHandler_(vector_handler), generator_(generator)
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
      TestOutcome testSolver(index_type nx,
                             index_type md,
                             index_type mc,
                             index_type H_nnz,
                             index_type Dx_nnz,
                             index_type Ds_nnz,
                             index_type J_nnz,
                             index_type Jd_nnz,
                             const std::string& H_file_name,
                             const std::string& Dx_file_name,
                             const std::string& Ds_file_name,
                             const std::string& J_file_name,
                             const std::string& Jd_file_name,
                             const std::string& rx_file_name,
                             const std::string& rs_file_name,
                             const std::string& ry_file_name,
                             const std::string& ryd_file_name,
                             real_type gamma)
      {
        constexpr double tol = 1e-12;
        
        std::ifstream H_file(H_file_name);
        std::ifstream Dx_file(Dx_file_name);
        std::ifstream Ds_file(Ds_file_name);
        std::ifstream J_file(J_file_name);
        std::ifstream Jd_file(Jd_file_name);
        std::ifstream rx_file(rx_file_name);
        std::ifstream rs_file(rs_file_name);
        std::ifstream ry_file(ry_file_name);
        std::ifstream ryd_file(ryd_file_name);

        // The .mtx file readers write into host accessible memory.
        // Load test data into HOST first, then sync to DEVICE for CUDA and HIP backends.
        matrix::Csr* H = new matrix::Csr(nx, nx, H_nnz);
        matrix::Csr* Dx = new matrix::Csr(nx, nx, Dx_nnz); // H and Dx need to be combined. The test matrices doesn't have Dx, so we'll reuse H as Dx
        matrix::Csr* Ds = new matrix::Csr(md, md, Ds_nnz);
        matrix::Csr* J = new matrix::Csr(mc, nx, J_nnz);
        matrix::Csr* Jd = new matrix::Csr(md, nx, Jd_nnz);
        H->allocateMatrixData(memory::HOST);
        Dx->allocateMatrixData(memory::HOST);
        Ds->allocateMatrixData(memory::HOST);
        J->allocateMatrixData(memory::HOST);
        Jd->allocateMatrixData(memory::HOST);
        if (memspace_ == memory::DEVICE)
        {
          H->syncData(memory::DEVICE);
          Dx->syncData(memory::DEVICE);
          Ds->syncData(memory::DEVICE);
          J->syncData(memory::DEVICE);
          Jd->syncData(memory::DEVICE);
        }

        // RHS vector blocks
        vector::Vector* rx = new vector::Vector(nx);
        vector::Vector* rs = new vector::Vector(md);
        vector::Vector* ry = new vector::Vector(mc);
        vector::Vector* ryd = new vector::Vector(md);
        rx->allocate(memory::HOST);
        rs->allocate(memory::HOST);
        ry->allocate(memory::HOST);
        ryd->allocate(memory::HOST);
        if (memspace_ == memory::DEVICE)
        {
          rx->syncData(memory::DEVICE);
          rs->syncData(memory::DEVICE);
          ry->syncData(memory::DEVICE);
          ryd->syncData(memory::DEVICE);
        }

        // LHS vector blocks
        vector::Vector* x = new vector::Vector(nx);
        vector::Vector* s = new vector::Vector(md);
        vector::Vector* y = new vector::Vector(mc);
        vector::Vector* yd = new vector::Vector(md);
        x->allocate(memspace_);
        s->allocate(memspace_);
        y->allocate(memspace_);
        yd->allocate(memspace_);

        hykkt::HyKKTSolver hykktSolver(nx, md, mc, memspace_);
        hykktSolver.setMatrixBlocks(H, Dx, Ds, J, Jd);
        hykktSolver.setRHSBlocks(rx, rs, ry, ryd);
        hykktSolver.setLHSPointers(x, s, y, yd);
        hykktSolver.readMatrixFiles(H_file, Dx_file, Ds_file, J_file, Jd_file, rx_file, rs_file, ry_file, ryd_file);
        hykktSolver.setGamma(gamma);
        hykktSolver.addHandlers(&matrixHandler_, &vectorHandler_);

        real_type error = hykktSolver.solve();

        TestStatus  status;
        std::string testname(__func__);
        index_type N = nx + mc + 2 * md;
        index_type nnz = H_nnz + Dx_nnz + Ds_nnz + J_nnz + Jd_nnz;
        testname += " N=" + std::to_string(N) + ", nnz =" + std::to_string(nnz);
        status *= validateResult(error, tol);

        delete H;
        delete Dx;
        delete Ds;
        delete J;
        delete Jd;
        delete rx;
        delete rs;
        delete ry;
        delete ryd;
        delete x;
        delete s;
        delete y;
        delete yd;

        return status.report(testname.c_str());
      }

    private:
      memory::MemorySpace memspace_;       ///< Memory space used by the test.
      MatrixHandler&      matrixHandler_; ///< Backend-specific matrix handler.
      VectorHandler&      vectorHandler_; ///< Backend-specific vector handler.
      std::mt19937&       generator_;      ///< C++ random number generator.

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
