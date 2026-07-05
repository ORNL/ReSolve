#pragma once

#include <algorithm>
#include <iomanip>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <random>

#include <resolve/Common.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
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
     * @class Tests for vector handler
     *
     */
    class VectorHandlerTests : TestBase
    {
    public:
      std::mt19937                generator_;
      VectorHandlerTests(ReSolve::VectorHandler& handler)
        : handler_(handler), generator_(ReSolve::constants::SEED)
      {
        if (handler_.getIsCudaEnabled() || handler_.getIsHipEnabled())
        {
          memspace_ = memory::DEVICE;
        }
        else
        {
          memspace_ = memory::HOST;
        }
      }

      virtual ~VectorHandlerTests()
      {
      }

      TestOutcome vectorHandlerConstructor()
      {
        TestStatus status;
        status.skipTest();

        return status.report(__func__);
      }

      TestOutcome amax(index_type N)
      {
        TestStatus status;
        status = true;

        vector::Vector x(N);

        real_type* data = new real_type[N];
        for (int i = 0; i < N; ++i)
        {
          data[i] = 0.1 * (real_type) i;
        }
        x.allocate(memspace_);
        x.copyFromExternal(data, memory::HOST, memspace_);

        real_type result = handler_.amax(&x, memspace_);
        real_type answer = static_cast<real_type>(N - 1) * 0.1;

        if (!isEqual(result, answer))
        {
          std::cout << "The result " << result << " is incorrect. "
                    << "Expected answer is " << answer << "\n";
          status *= false;
        }

        delete[] data;
        return status.report(__func__);
      }

      TestOutcome axpy(index_type N)
      {
        TestStatus status;

        vector::Vector x(N);
        vector::Vector y(N);

        x.allocate(memspace_);
        y.allocate(memspace_);

        x.setToConst(3.0, memspace_);
        y.setToConst(1.0, memspace_);

        real_type alpha = ReSolve::constants::HALF;

        // the result is a vector with y[i] = 2.5 forall i;
        handler_.axpy(alpha, &x, &y, memspace_);
        status *= verifyAnswer(y, 2.5);

        return status.report(__func__);
      }

      TestOutcome dot(index_type N)
      {
        TestStatus status;
        status = true;

        vector::Vector x(N);
        vector::Vector y(N);

        x.allocate(memspace_);
        y.allocate(memspace_);

        x.setToConst(0.25, memspace_);
        y.setToConst(4.0, memspace_);

        // the answer is N
        real_type answer = static_cast<real_type>(N);
        real_type result = handler_.dot(&x, &y, memspace_);

        if (!isEqual(result, answer))
        {
          std::cout << "The result " << result << " is incorrect. "
                    << "Expected answer is " << answer << "\n";
          status *= false;
        }

        return status.report(__func__);
      }

      TestOutcome scal(index_type N)
      {
        TestStatus status;

        vector::Vector x(N);

        x.allocate(memspace_);

        x.setToConst(1.25, memspace_);

        real_type alpha = 3.5;

        // the answer is x[i] = 4.375;
        real_type answer = 4.375;
        handler_.scal(alpha, &x, memspace_);
        status *= verifyAnswer(x, answer);

        return status.report(__func__);
      }

      TestOutcome axpyMulti(index_type N, index_type K)
      {
        TestStatus status;

        vector::Vector x(N, K);
        vector::Vector y(N);
        vector::Vector alpha(K);
        ;

        x.allocate(memspace_);
        y.allocate(memspace_);
        alpha.allocate(memspace_);

        alpha.setToConst(-1.0, memspace_);
        y.setToConst(2.0, memspace_);

        for (int ii = 0; ii < K; ++ii)
        {
          real_type c;
          if (ii % 2 == 0)
          {
            c = -1.0;
          }
          else
          {
            c = ReSolve::constants::HALF;
          }
          x.setToConst(ii, c, memspace_);
        }

        index_type r   = K % 2;
        real_type  res = (real_type) ((floor((real_type) K / 2.0) + r) * 1.0 + floor((real_type) K / 2.0) * (-0.5));

        handler_.axpyMulti(N, &alpha, K, &x, &y, memspace_);
        status *= verifyAnswer(y, 2.0 - res);

        return status.report(__func__);
      }

      TestOutcome massDot(index_type N, index_type K)
      {
        TestStatus status;

        vector::Vector x(N, K);
        vector::Vector y(N, 2);
        vector::Vector res(K, 2);
        x.allocate(memspace_);
        y.allocate(memspace_);
        res.allocate(memspace_);

        x.setToConst(1.0, memspace_);
        y.setToConst(-1.0, memspace_);
        handler_.dot2Multi(N, &x, K, &y, &res, memspace_);

        status *= verifyAnswer(res, (-1.0) * (real_type) N);

        return status.report(__func__);
      }

      TestOutcome gemv(index_type N, index_type K)
      {
        TestStatus status;

        vector::Vector V(N, K);
        vector::Vector yN(K); ///< For the test with NO TRANSPOSE
        vector::Vector xN(N);
        vector::Vector yT(N); ///< for the test with TRANSPOSE
        vector::Vector xT(K);

        V.allocate(memspace_);
        yN.allocate(memspace_);
        xN.allocate(memspace_);
        yT.allocate(memspace_);
        xT.allocate(memspace_);

        V.setToConst(1.0, memspace_);
        yN.setToConst(-1.0, memspace_);
        xN.setToConst(ReSolve::constants::HALF, memspace_);
        yT.setToConst(-1.0, memspace_);
        xT.setToConst(ReSolve::constants::HALF, memspace_);

        real_type alpha = -1.0;
        real_type beta  = 1.0;
        handler_.gemv('N', K, alpha, beta, &V, &yN, &xN, memspace_);
        status *= verifyAnswer(xN, static_cast<real_type>(K) + ReSolve::constants::HALF);
        handler_.gemv('T', K, alpha, beta, &V, &yT, &xT, memspace_);
        status *= verifyAnswer(xT, static_cast<real_type>(N) + ReSolve::constants::HALF);

        return status.report(__func__);
      }

      TestOutcome gemm(index_type N, index_type M, index_type K)
      {
        TestStatus status;

        vector::Vector A_N(M, K); ///< For the test with NO TRANSPOSE
        vector::Vector B_N(K, N);
        vector::Vector A_T(K, M); ///< for the test with TRANSPOSE
        vector::Vector B_T(N, K);
        vector::Vector C_NN(M, N);
        vector::Vector C_NT(M, N);
        vector::Vector C_TN(M, N);
        vector::Vector C_TT(M, N);

        A_N.allocate(memspace_);
        B_N.allocate(memspace_);
        A_T.allocate(memspace_);
        B_T.allocate(memspace_);
        C_NN.allocate(memspace_);
        C_NT.allocate(memspace_);
        C_TN.allocate(memspace_);
        C_TT.allocate(memspace_);

        A_N.setToConst(1.0, memspace_);
        B_N.setToConst(-1.0, memspace_);
        A_T.setToConst(1.0, memspace_);
        B_T.setToConst(-1.0, memspace_);
        C_NN.setToConst(ReSolve::constants::HALF, memspace_);
        C_NT.setToConst(ReSolve::constants::HALF, memspace_);
        C_TN.setToConst(ReSolve::constants::HALF, memspace_);
        C_TT.setToConst(ReSolve::constants::HALF, memspace_);

        real_type alpha = -1.0;
        real_type beta  = 1.0;
        handler_.gemm('N', 'N', alpha, beta, &A_N, &B_N, &C_NN, memspace_);
        status *= verifyAnswer(C_NN, static_cast<real_type>(K) + ReSolve::constants::HALF);
        handler_.gemm('N', 'T', alpha, beta, &A_N, &B_T, &C_NT, memspace_);
        status *= verifyAnswer(C_NT, static_cast<real_type>(K) + ReSolve::constants::HALF);
        handler_.gemm('T', 'N', alpha, beta, &A_T, &B_N, &C_TN, memspace_);
        status *= verifyAnswer(C_TN, static_cast<real_type>(K) + ReSolve::constants::HALF);
        handler_.gemm('T', 'T', alpha, beta, &A_T, &B_T, &C_TT, memspace_);
        status *= verifyAnswer(C_TT, static_cast<real_type>(K) + ReSolve::constants::HALF);

        return status.report(__func__);
      }

      vector::Vector* randomVector(index_type N, index_type K)
      {
        vector::Vector* v = new vector::Vector(N, K);
        v->allocateAll(memspace_);
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        for (index_type i = 0; i < K; ++i)
        {
          for (index_type j = 0; j < N; ++j)
          {
            v->getData(i, memory::HOST)[j] = distribution(generator_);
          }
        }
        v->setDataUpdated(memory::HOST);
        if (memspace_ == memory::DEVICE)
        {
          v->syncData(memory::DEVICE);
        }
        return v;
      }
      
      TestOutcome gemmSpeedTest(index_type N, index_type K)
      {
        TestStatus status;
        
        vector::Vector* V = randomVector(N, N);
        vector::Vector* W = randomVector(N, K);
        vector::Vector* result = new vector::Vector(N, K);
        result->allocate(memspace_);

        real_type alpha = 1.0;
        real_type beta  = 0.0;

        // if (K == 1) {
        //   auto start = std::chrono::steady_clock::now();
        //   handler_.dot(V, W, memspace_);
        //   auto end = std::chrono::steady_clock::now();
        //   std::chrono::duration<double, std::milli> elapsed = (end - start);
        //   printf("n = %d, k = %d, time = %f\n", N, K, elapsed.count());
        // } else {
        for (int i=0; i < 8; i++) {
          auto start = std::chrono::steady_clock::now();
          handler_.gemm('N', 'N', alpha, beta, V, W, result, memspace_);
          auto end = std::chrono::steady_clock::now();
          std::chrono::duration<double, std::milli> elapsed = (end - start);
          deviceSynchronize();
          printf("n = %d, k = %d, time = %f\n", N, K, elapsed.count());
          
          
          start = std::chrono::steady_clock::now();
          handler_.gemm('N', 'N', alpha, beta, V, result, W, memspace_);
          end = std::chrono::steady_clock::now();
          elapsed = (end - start);
          deviceSynchronize();
          printf("n = %d, k = %d, time = %f\n", N, K, elapsed.count());
        }
        // }

        delete V;
        delete W;
        delete result;

        return status.report(__func__);
      }

      TestOutcome scale(index_type N)
      {
        TestStatus status;

        vector::Vector diag(N);
        vector::Vector vec(N);

        // diag[i] = i, vec[i] = 3.0
        // expected result vec[i] = i * 3.0
        diag.allocate(memspace_);
        vec.allocateAll(memspace_);

        vec.setToConst(3.0, memspace_);

        auto diag_data = std::unique_ptr<real_type[]>(new real_type[N]);
        for (size_t i = 0; i < static_cast<size_t>(N); ++i)
        {
          diag_data[i] = (real_type) (i + 1);
        }
        diag.copyFromExternal(diag_data.get(), memory::HOST, memspace_);

        handler_.scal(&diag, &vec, memspace_);

        if (memspace_ == memory::DEVICE)
        {
          vec.syncData(memory::HOST);
        }

        for (index_type i = 0; i < N; ++i)
        {
          if (!isEqual(vec.getData(memory::HOST)[i], (real_type) (i + 1) * 3.0))
          {
            std::cout << "Solution vector element vec[" << i << "] = " << vec.getData(memory::HOST)[i]
                      << ", expected: " << (real_type) (i + 1) * 3.0 << "\n";
            status *= false;
            break;
          }
        }

        return status.report(__func__);
      }

      TestOutcome diagSolve(index_type N)
      {
        TestStatus status;

        vector::Vector diag(N);
        vector::Vector vec(N);

        // diag[i] = i + 1, vec[i] = 3.0
        // expected result vec[i] = 3.0 / (i + 1)
        diag.allocate(memspace_);
        vec.allocate(memspace_);

        vec.setToConst(3.0, memspace_);

        auto diag_data = std::unique_ptr<real_type[]>(new real_type[N]);
        for (size_t i = 0; i < static_cast<size_t>(N); ++i)
        {
          diag_data[i] = (real_type) (i + 1);
        }
        diag.copyFromExternal(diag_data.get(), memory::HOST, memspace_);

        handler_.diagSolve(&diag, &vec, memspace_);

        if (memspace_ == memory::DEVICE)
        {
          vec.syncData(memory::HOST);
        }

        for (index_type i = 0; i < N; ++i)
        {
          if (!isEqual(vec.getData(memory::HOST)[i], (real_type) 3.0 / (i + 1)))
          {
            std::cout << "Solution vector element vec[" << i << "] = " << vec.getData(memory::HOST)[i]
                      << ", expected: " << (real_type) 3.0 / (i + 1) << "\n";
            status *= false;
            break;
          }
        }

        return status.report(__func__);
      }

      TestOutcome max(index_type N)
      {
        TestStatus status;

        vector::Vector x(N);
        vector::Vector y(N);
        vector::Vector z(N);

        x.allocate(memspace_);
        y.allocateAll(memspace_);
        z.allocateAll(memspace_);

        auto x_data = std::unique_ptr<real_type[]>(new real_type[N]);
        auto y_data = std::unique_ptr<real_type[]>(new real_type[N]);
        for (size_t i = 0; i < static_cast<size_t>(N); ++i)
        {
          if (i % 3 == 0)
          {
            x_data[i] = (real_type) (i + 1);
            y_data[i] = (real_type) i * 0.5;
          }
          else
          {
            x_data[i] = -(real_type) (i + 1);
            y_data[i] = (real_type) (i + 1);
          }
        }
        x.copyFromExternal(x_data.get(), memory::HOST, memspace_);
        y.copyFromExternal(y_data.get(), memory::HOST, memspace_);

        handler_.max(&x, &y, &z, memspace_);
        handler_.max(&x, &y, &y, memspace_);

        if (memspace_ == memory::DEVICE)
        {
          y.syncData(memory::HOST);
          z.syncData(memory::HOST);
        }

        for (index_type i = 0; i < N; ++i)
        {
          if (!isEqual(y.getData(memory::HOST)[i], (real_type) (i + 1)))
          {
            std::cout << "Solution vector element y[" << i << "] = " << y.getData(memory::HOST)[i]
                      << ", expected: " << (real_type) (i + 1) << "\n";
            status *= false;
            break;
          }

          if (!isEqual(z.getData(memory::HOST)[i], (real_type) (i + 1)))
          {
            std::cout << "Solution vector element z[" << i << "] = " << z.getData(memory::HOST)[i]
                      << ", expected: " << (real_type) (i + 1) << "\n";
            status *= false;
            break;
          }
        }

        return status.report(__func__);
      }

      TestOutcome abs(index_type N)
      {
        TestStatus status;

        vector::Vector x(N);
        vector::Vector y(N);

        x.allocateAll(memspace_);
        y.allocateAll(memspace_);

        auto x_data = std::unique_ptr<real_type[]>(new real_type[N]);
        for (size_t i = 0; i < static_cast<size_t>(N); ++i)
        {
          if (i % 3 == 0)
          {
            x_data[i] = -(real_type) i;
          }
          else
          {
            x_data[i] = (real_type) i;
          }
        }
        x.copyFromExternal(x_data.get(), memory::HOST, memspace_);

        handler_.abs(&x, &y, memspace_);
        handler_.abs(&x, &x, memspace_);

        if (memspace_ == memory::DEVICE)
        {
          x.syncData(memory::HOST);
          y.syncData(memory::HOST);
        }

        for (index_type i = 0; i < N; ++i)
        {
          if (!isEqual(x.getData(memory::HOST)[i], (real_type) i))
          {
            std::cout << "Solution vector element x[" << i << "] = " << x.getData(memory::HOST)[i]
                      << ", expected: " << (real_type) i << "\n";
            status *= false;
            break;
          }

          if (!isEqual(y.getData(memory::HOST)[i], (real_type) i))
          {
            std::cout << "Solution vector element y[" << i << "] = " << y.getData(memory::HOST)[i]
                      << ", expected: " << (real_type) i << "\n";
            status *= false;
            break;
          }
        }

        return status.report(__func__);
      }

    private:
      ReSolve::VectorHandler&      handler_;
      ReSolve::memory::MemorySpace memspace_{memory::HOST};

      // we can verify through norm but that would defeat the purpose of testing vector handler ...
      bool verifyAnswer(vector::Vector& x, real_type answer)
      {
        bool status = true;

        if (memspace_ == memory::DEVICE)
        {
          x.syncData(memory::HOST);
        }

        for (index_type i = 0; i < x.getSize(); ++i)
        {
          // std::cout << x->getData("cpu")[i] << "\n";
          if (!isEqual(x.getData(memory::HOST)[i], answer))
          {
            std::cout << std::setprecision(16);
            status = false;
            std::cout << "Solution vector element x[" << i << "] = " << x.getData(memory::HOST)[i]
                      << ", expected: " << answer << "\n";
            break;
          }
        }
        return status;
      }
    }; // class
  } // namespace tests
} // namespace ReSolve
