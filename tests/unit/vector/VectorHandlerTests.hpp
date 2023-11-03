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
        VectorHandlerTests(std::string memspace) : memspace_(memspace) 
        {
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

        TestOutcome axpy(index_type N)
        {
          TestStatus status;

          ReSolve::memory::MemorySpace ms;
          if (memspace_ == "cpu")
            ms = memory::HOST;
          else
            ms = memory::DEVICE;

          ReSolve::VectorHandler* handler = createVectorHandler();

          vector::Vector* x = new vector::Vector(N);
          vector::Vector* y = new vector::Vector(N);

          x->allocate(ms);
          y->allocate(ms);

          x->setToConst(3.0, ms);
          y->setToConst(1.0, ms);

          real_type alpha = 0.5;
          //the result is a vector with y[i] = 2.5;          
          handler->axpy(&alpha, x, y, memspace_);
          status *= verifyAnswer(y, 2.5, memspace_);

          delete handler;
          delete x;
          delete y; 

          return status.report(__func__);
        }    

        TestOutcome dot(index_type N)
        {
          TestStatus status;

          ReSolve::memory::MemorySpace ms;
          if (memspace_ == "cpu")
            ms = memory::HOST;
          else
            ms = memory::DEVICE;

          ReSolve::VectorHandler* handler = createVectorHandler();

          vector::Vector* x = new vector::Vector(N);
          vector::Vector* y = new vector::Vector(N);

          x->allocate(ms);
          y->allocate(ms);

          x->setToConst(0.25, ms);
          y->setToConst(4.0, ms);
          real_type ans;
          //the result is N
          ans = handler->dot(x, y, memspace_);

          bool st = true;;
          if (ans != (real_type) N) {
            st = false;
            printf("the wrong answer is %f expecting %f \n", ans, (real_type) N);
          } 
          status *= st;

          delete handler;
          delete x;
          delete y; 

          return status.report(__func__);
        }    

        TestOutcome scal(index_type N)
        {
          TestStatus status;

          ReSolve::memory::MemorySpace ms;
          if (memspace_ == "cpu")
            ms = memory::HOST;
          else
            ms = memory::DEVICE;

          ReSolve::VectorHandler* handler = createVectorHandler();

          vector::Vector* x =  new vector::Vector(N);

          x->allocate(ms);

          x->setToConst(1.25, ms);

          real_type alpha = 3.5;

          //the answer is x[i] = 4.375;         
          handler->scal(&alpha, x, memspace_);
          status *= verifyAnswer(x, 4.375, memspace_);

          delete handler;
          delete x;

          return status.report(__func__);
        }    

        TestOutcome massAxpy(index_type N, index_type K)
        {
          TestStatus status;

          ReSolve::memory::MemorySpace ms;
          if (memspace_ == "cpu")
            ms = memory::HOST;
          else
            ms = memory::DEVICE;

          ReSolve::VectorHandler* handler = createVectorHandler();
          
          vector::Vector* x =  new vector::Vector(N, K);
          vector::Vector* y =  new vector::Vector(N);
          vector::Vector* alpha = new vector::Vector(K);;
          x->allocate(ms);
          y->allocate(ms);
          alpha->allocate(ms);

          y->setToConst(2.0, ms);
          alpha->setToConst(-1.0, ms);
          for (int ii = 0; ii < K; ++ii) {
            real_type c;
            if (ii % 2 == 0) {
              c = -1.0;
            } else {
              c = 0.5;
            }
            x->setToConst(ii, c, ms);
          }

          index_type r = K % 2;
          real_type res = (real_type) ((floor((real_type) K / 2.0) + r) * 1.0 + floor((real_type) K / 2.0) * (-0.5));

          handler->massAxpy(N, alpha, K, x, y, memspace_);
          status *= verifyAnswer(y, 2.0 - res, memspace_);
         
          delete handler;
          delete x;
          delete y;
          delete alpha;

          return status.report(__func__);
        }    

        TestOutcome massDot(index_type N, index_type K)
        {
          TestStatus status;

          ReSolve::memory::MemorySpace ms;
          if (memspace_ == "cpu")
            ms = memory::HOST;
          else
            ms = memory::DEVICE;

          ReSolve::VectorHandler* handler = createVectorHandler();
          
          vector::Vector* x =  new vector::Vector(N, K);
          vector::Vector* y =  new vector::Vector(N, 2);
          vector::Vector* res = new vector::Vector(K, 2);
          x->allocate(ms);
          y->allocate(ms);
          res->allocate(ms);
          
          x->setToConst(1.0, ms);
          y->setToConst(-1.0, ms);
          handler->massDot2Vec(N, x, K, y, res, memspace_);
          
          status *= verifyAnswer(res, (-1.0) * (real_type) N, memspace_);

          delete handler;
          delete x;
          delete y;
          delete res;
          return status.report(__func__);
        }    

        TestOutcome gemv(index_type N,  index_type K)
        {
          TestStatus status;

          ReSolve::memory::MemorySpace ms;
          if (memspace_ == "cpu")
            ms = memory::HOST;
          else
            ms = memory::DEVICE;

          ReSolve::VectorHandler* handler = createVectorHandler();
          vector::Vector* V = new vector::Vector(N, K);
          // for the test with NO TRANSPOSE
          vector::Vector* yN = new vector::Vector(K); 
          vector::Vector* xN = new vector::Vector(N);
          // for the test with TRANSPOSE
          vector::Vector* yT = new vector::Vector(N);
          vector::Vector* xT = new vector::Vector(K);
          
          V->allocate(ms);
          yN->allocate(ms);
          xN->allocate(ms);
          yT->allocate(ms);
          xT->allocate(ms);

          V->setToConst(1.0, ms);
          yN->setToConst(-1.0, ms);
          xN->setToConst(.5, ms);
          yT->setToConst(-1.0, ms);
          xT->setToConst(.5, ms);
          
          real_type alpha = -1.0;
          real_type beta = 1.0;
          handler->gemv("N", N, K, &alpha, &beta, V, yN, xN, memspace_);
          status *= verifyAnswer(xN, (real_type) (K) + 0.5, memspace_);
          handler->gemv("T", N, K, &alpha, &beta, V, yT, xT, memspace_);
          status *= verifyAnswer(xT, (real_type) (N) + 0.5, memspace_);

          return status.report(__func__);
        }    

      private:
        std::string memspace_{"cpu"};

        ReSolve::VectorHandler* createVectorHandler()
        {
          if (memspace_ == "cpu") {
            LinAlgWorkspaceCpu* workpsace = new LinAlgWorkspaceCpu();
            return new VectorHandler(workpsace);
#ifdef RESOLVE_USE_CUDA
          } else if (memspace_ == "cuda") {
            LinAlgWorkspaceCUDA* workspace = new LinAlgWorkspaceCUDA();
            workspace->initializeHandles();
            return new VectorHandler(workspace);
#endif
#ifdef RESOLVE_USE_HIP
          } else if (memspace_ == "hip") {
            LinAlgWorkspaceHIP* workspace = new LinAlgWorkspaceHIP();
            workspace->initializeHandles();
            return new VectorHandler(workspace);
#endif
          } else {
            std::cout << "ReSolve not built with support for memory space " << memspace_ << "\n";
          }
          return nullptr;
        }

        // we can verify through norm but that would defeat the purpose of testing vector handler ...
        bool verifyAnswer(vector::Vector* x, real_type answer, std::string memspace)
        {
          bool status = true;
          if (memspace != "cpu") {
            x->copyData(memory::DEVICE, memory::HOST);
          }

          for (index_type i = 0; i < x->getSize(); ++i) {
            // std::cout << x->getData("cpu")[i] << "\n";
            if (!isEqual(x->getData(memory::HOST)[i], answer)) {
              std::cout << std::setprecision(16);
              status = false;
              std::cout << "Solution vector element x[" << i << "] = " << x->getData(memory::HOST)[i]
                << ", expected: " << answer << "\n";
              break; 
            }
          }
          return status;
        }
    };//class
  }
}

