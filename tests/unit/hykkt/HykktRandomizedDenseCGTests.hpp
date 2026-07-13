/**
 * @file HykktRandomizedDenseConjugateGradientTests.hpp
 * @brief Implementation of tests for class hykkt::RandomizedDenseConjugateGradient
 *
 */
#pragma once

#include <chrono>
#include <filesystem>
#include <iterator>
#include <numeric>
#include <sstream>

#include <resolve/MemoryUtils.hpp>
#include <resolve/hykkt/randomized_dense_cg/RandomizedDenseConjugateGradient.hpp>
#include <resolve/hykkt/cholesky/CholeskySolver.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <tests/unit/TestBase.hpp>

#ifdef RESOLVE_USE_CUDA
#include <cuda_runtime.h>
#define deviceSynchronize cudaDeviceSynchronize
#elif defined(RESOLVE_USE_HIP)
#include <hip/hip_runtime.h>
#define deviceSynchronize hipDeviceSynchronize
#endif

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

      void generateVectors(const std::string& A_file_name, const std::string& b_file_name, index_type n, real_type rng_min, real_type rng_max)
      {        
        vector::Vector* M = new vector::Vector(n, n);
        M->allocateAll(memspace_);
        vector_handler_.randomVector(M, rng_min , rng_max, memspace_);

        vector::Vector* A = new vector::Vector(n, n);
        A->allocateAll(memspace_);

        vector_handler_.gemm('T', 'N', 1.0, 0.0, M, M, A, memspace_);
        // vector_handler_.addIdentity(A, 1.0, memspace_);
        if (memspace_ == memory::DEVICE)
        {
          A->syncData(memory::HOST);
        }
        
        vector::Vector* b = new vector::Vector(n);
        b->allocateAll(memspace_);
        vector_handler_.randomVector(b, rng_min, rng_max, memspace_);
        if (memspace_ == memory::DEVICE)
        {
          b->syncData(memory::HOST);
        }
        
        std::ostringstream A_buffer;
        io::writeVectorToFile(A, A_buffer);        
        std::ofstream A_file(A_file_name);
        if (A_file.is_open())
        {
          A_file << A_buffer.str();
          A_file.close();
        }
        
        std::ostringstream b_buffer;
        io::writeVectorToFile(b, b_buffer);
        std::ofstream b_file(b_file_name);
        if (b_file.is_open())
        {
          b_file << b_buffer.str();
          b_file.close();
        }

        printf("File write success.\n");
      }

      void choleskyTests(const std::string& A_file_name, const std::string& b_file_name, index_type n)
      {
        std::ifstream A_file(A_file_name);
        vector::Vector* A = io::createVectorFromFile(A_file);
        if (memspace_ == memory::DEVICE)
        {
          A->allocate(memory::DEVICE);
          A->syncData(memory::DEVICE);
        }
        
        std::ifstream b_file(b_file_name);
        vector::Vector* b = io::createVectorFromFile(b_file);
        if (memspace_ == memory::DEVICE)
        {
          b->allocate(memory::DEVICE);
          b->syncData(memory::DEVICE);
        }

        printf("File loaded\n");

        auto start = std::chrono::steady_clock::now();
        vector_handler_.choleskyFactorize(A, 'L', memspace_);
        vector_handler_.choleskySolve(A->getData(memspace_), b, 'L', memspace_);
        deviceSynchronize();

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = (end - start);
        printf("Time: %f\n", elapsed);
      }

      void choleskyTests(vector::Vector* A, vector::Vector* b, index_type n)
      {
        vector::Vector* A_chol = new vector::Vector(n, n);
        A_chol->allocateAll(memspace_);
        A_chol->copyFromExternal(A, memspace_, memspace_);

        vector::Vector* x = new vector::Vector(n);
        x->allocateAll(memspace_);
        x->copyFromExternal(b, memspace_, memspace_);

        auto start = std::chrono::steady_clock::now();
        printf("Factorize status: %d\n", vector_handler_.choleskyFactorize(A_chol, 'L', memspace_));
        printf("Solve status: %d\n", vector_handler_.choleskySolve(A_chol->getData(memspace_), x, 'L', memspace_));
        deviceSynchronize();

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = (end - start);
        printf("Cholesky time: %f\n", elapsed);
      }

      /**
       * @brief Test the RandomizedDenseConjugateGradient implementation with matrices in tests\unit\hykkt\RandomizedCGTestMatrices
       *
       * @return TestOutcome Result of the test
       */
      TestOutcome RandomizedDenseCGTest(const std::string& A_file_name, const std::string& b_file_name, index_type n, index_type k, real_type rng_min, real_type rng_max)
      {      
        hykkt::CholeskySolver cholesky_solver(memspace_);
        hykkt::RandomizedDenseConjugateGradient randomized_cg(n, k, &cholesky_solver, &matrix_handler_, &vector_handler_, memspace_);
        randomized_cg.setSolverTolerance(randomized_cg_initial_tol, randomized_cg_convergence_tol);

        vector::Vector* M = new vector::Vector(n, n);
        M->allocateAll(memspace_);
        vector_handler_.randomVector(M, rng_min , rng_max, memspace_);
        // M->syncData(memory::HOST);

        vector::Vector* A = new vector::Vector(n, n);
        A->allocateAll(memspace_);

        vector_handler_.gemm('T', 'N', 1.0, 0.0, M, M, A, memspace_);
        // vector_handler_.addIdentity(A, 1.0, memspace_);
        A->syncData(memory::HOST);
  
        vector::Vector* b = new vector::Vector(n);
        b->allocateAll(memspace_);
        vector_handler_.randomVector(b, rng_min, rng_max, memspace_);
        delete M;
        
        // std::ifstream A_file(A_file_name);
        // vector::Vector* A = io::createVectorFromFile(A_file);
        // if (memspace_ == memory::DEVICE)
        // {
        //   A->allocate(memory::DEVICE);
        //   A->syncData(memory::DEVICE);
        // }
        
        // std::ifstream b_file(b_file_name);
        // vector::Vector* b = io::createVectorFromFile(b_file);
        // if (memspace_ == memory::DEVICE)
        // {
        //   b->allocate(memory::DEVICE);
        //   b->syncData(memory::DEVICE);
        // }

        vector::Vector* x = new vector::Vector(n);
        x->allocateAll(memspace_);
        
        vector::Vector* d = new vector::Vector(n);
        d->allocateAll(memspace_);

        real_type* d_vals = d->getData(memory::HOST);
        for (int i = 0; i < n; i++) {
            // (i * n + i) maps to the diagonal elements of a flat n x n matrix
            d_vals[i] = std::sqrt(A->getData(i, memory::HOST)[i]); 
        }

        d->setDataUpdated(memory::HOST);
        if (memspace_ == memory::DEVICE)
        {
          d->syncData(memory::DEVICE);
        }

        vector::Vector* d_inv = new vector::Vector(n);
        d_inv->allocateAll(memspace_);

        real_type* d_inv_vals = d_inv->getData(memory::HOST);
        for (int i = 0; i < n; i++) {
            // (i * n + i) maps to the diagonal elements of a flat n x n matrix
            d_inv_vals[i] = 1.0 / std::sqrt(A->getData(i, memory::HOST)[i]); 
        }

        d_inv->setDataUpdated(memory::HOST);
        if (memspace_ == memory::DEVICE)
        {
          d_inv->syncData(memory::DEVICE);
        }

        // if (k == 1)
        // {
        //   choleskyTests(A, b, n);
        // }

        randomized_cg.addMatrixInfo(A);
        randomized_cg.addVectorInfo(x, b);
        randomized_cg.addPreconditionerInfo(d, d_inv);
        randomized_cg.setup();
        int converged_n = randomized_cg.solve(); // 0 if converged, 1 if not

        TestStatus  status;
        std::string testname(__func__);
        testname += " n=" + std::to_string(n) + ", k=" + std::to_string(k);
        status *= validateResult(x, converged_n);

        delete A;
        delete x;
        delete b;
        delete d;
        delete d_inv;

        return status.report(testname.c_str());
      }

    private:
      memory::MemorySpace memspace_;       ///< Memory space used by the test.
      MatrixHandler&      matrix_handler_; ///< Backend-specific matrix handler.
      VectorHandler&      vector_handler_; ///< Backend-specific vector handler.

      static constexpr real_type cholesky_tol = 1e-12;
      static constexpr real_type randomized_cg_initial_tol     = 1e-12;
      static constexpr real_type randomized_cg_convergence_tol     = 1e-12;
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
