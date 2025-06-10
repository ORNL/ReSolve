#pragma once
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <tests/unit/TestBase.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

namespace ReSolve {
  namespace tests {
    /**
     * @class Tests for vector handler
     *
     */
    class VectorHandlerTests : TestBase
    {
      public:
        VectorHandlerTests(ReSolve::VectorHandler& handler) : handler_(handler)
        {
          if (handler_.getIsCudaEnabled() || handler_.getIsHipEnabled()) {
            memspace_ = memory::DEVICE;
          } else {
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

        TestOutcome infNorm(index_type N)
        {
          TestStatus status;
          status = true;

          vector::Vector x(N);

          real_type* data = new real_type[N];
          for (int i = 0; i < N; ++i) {
            data[i] = 0.1 * (real_type) i;
          }
          x.copyDataFrom(data, memory::HOST, memspace_);

          real_type result = handler_.infNorm(&x, memspace_);
          real_type answer = static_cast<real_type>(N - 1) * 0.1;

          if (!isEqual(result, answer)) {
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

          real_type alpha = 0.5;

          //the result is a vector with y[i] = 2.5 forall i;
          handler_.axpy(&alpha, &x, &y, memspace_);
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

          if (!isEqual(result, answer)) {
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

          //the answer is x[i] = 4.375;
          real_type answer = 4.375;
          handler_.scal(&alpha, &x, memspace_);
          status *= verifyAnswer(x, answer);

          return status.report(__func__);
        }

        TestOutcome massAxpy(index_type N, index_type K)
        {
          TestStatus status;

          vector::Vector x(N, K);
          vector::Vector y(N);
          vector::Vector alpha(K);;

          x.allocate(memspace_);
          y.allocate(memspace_);
          alpha.allocate(memspace_);

          alpha.setToConst(-1.0, memspace_);
          y.setToConst(2.0, memspace_);

          for (int ii = 0; ii < K; ++ii) {
            real_type c;
            if (ii % 2 == 0) {
              c = -1.0;
            } else {
              c = 0.5;
            }
            x.setToConst(ii, c, memspace_);
          }

          index_type r = K % 2;
          real_type res = (real_type) ((floor((real_type) K / 2.0) + r) * 1.0 + floor((real_type) K / 2.0) * (-0.5));

          handler_.massAxpy(N, &alpha, K, &x, &y, memspace_);
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
          handler_.massDot2Vec(N, &x, K, &y, &res, memspace_);

          status *= verifyAnswer(res, (-1.0) * (real_type) N);

          return status.report(__func__);
        }

        TestOutcome gemv(index_type N,  index_type K)
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
          xN.setToConst(.5, memspace_);
          yT.setToConst(-1.0, memspace_);
          xT.setToConst(.5, memspace_);

          real_type alpha = -1.0;
          real_type beta = 1.0;
          handler_.gemv('N', N, K, &alpha, &beta, &V, &yN, &xN, memspace_);
          status *= verifyAnswer(xN, static_cast<real_type>(K) + 0.5);
          handler_.gemv('T', N, K, &alpha, &beta, &V, &yT, &xT, memspace_);
          status *= verifyAnswer(xT, static_cast<real_type>(N) + 0.5);

          return status.report(__func__);
        }

        TestOutcome vectorScale(index_type N)
        {
          TestStatus status;

          vector::Vector diag(N);
          vector::Vector vec(N);

          // diag[i] = i, vec[i] = 3.0
          // expected result vec[i] = i * 3.0
          diag.allocate(memspace_);
          vec.allocate(memspace_);

          vec.setToConst(3.0, memspace_);

          real_type* diag_data = new real_type[N];
          for (index_type i = 1; i <= N; ++i) {
            diag_data[i] = (real_type)i;
          }
          diag.copyDataFrom(diag_data, memory::HOST, memspace_)
          
          handler_.vectorScale(&diag, &vec, memspace_);

          vec.syncData(memory::HOST);

          for (index_type i = 1; i <= N; ++i) {
            if (!isEqual(vec.getData(memory::HOST)[i], (real_type)i * 3.0)) {
              std::cout << "Solution vector element vec[" << i << "] = " << vec.getData(memory::HOST)[i]
                << ", expected: " << (real_type)i * 3.0 << "\n";
              status *= false;
              break; 
            }
          }

          return status.report(__func__);
        }

      private:
        ReSolve::VectorHandler& handler_;
        ReSolve::memory::MemorySpace memspace_{memory::HOST};

        // we can verify through norm but that would defeat the purpose of testing vector handler ...
        bool verifyAnswer(vector::Vector& x, real_type answer)
        {
          bool status = true;

          if (memspace_ == memory::DEVICE) {
            x.syncData(memory::HOST);
          }

          for (index_type i = 0; i < x.getSize(); ++i) {
            // std::cout << x->getData("cpu")[i] << "\n";
            if (!isEqual(x.getData(memory::HOST)[i], answer)) {
              std::cout << std::setprecision(16);
              status = false;
              std::cout << "Solution vector element x[" << i << "] = " << x.getData(memory::HOST)[i]
                << ", expected: " << answer << "\n";
              break;
            }
          }
          return status;
        }
    };//class
  } // namespace tests
} //namespace ReSolve
