#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <tests/unit/TestBase.hpp>

namespace ReSolve { 
  namespace tests {
    /** @class Tests for vector handler
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

          ReSolve::LinAlgWorkspace* workspace = createLinAlgWorkspace(memspace_);
          ReSolve::VectorHandler handler(workspace);

          vector::Vector* x = new vector::Vector(N);
          vector::Vector* y = new vector::Vector(N);

          x->allocate(memspace_);
          y->allocate(memspace_);

          x->setToConst(3.0, memspace_);
          y->setToConst(1.0, memspace_);

          real_type alpha = 0.5;
          //the result is a vector with y[i] = 2.5;          
          handler.axpy(&alpha, x, y, memspace_);
          status *= verifyAnswer(y, 2.5, memspace_);

          delete workspace;
          delete x;
          delete y; 

          return status.report(__func__);
        }    

        TestOutcome dot(index_type N)
        {
          TestStatus status;

          ReSolve::LinAlgWorkspace* workspace = createLinAlgWorkspace(memspace_);
          ReSolve::VectorHandler handler(workspace);

          vector::Vector* x = new vector::Vector(N);
          vector::Vector* y = new vector::Vector(N);

          x->allocate(memspace_);
          y->allocate(memspace_);

          x->setToConst(0.25, memspace_);
          y->setToConst(4.0, memspace_);
          real_type ans;
          //the result is N
          ans = handler.dot(x, y, memspace_);

          bool st = true;;
          if (ans != (real_type) N) {
            st = false;
            printf("the wrong answer is %f expecting %f \n", ans, (real_type) N);
          } 
          status *= st;

          delete workspace;
          delete x;
          delete y; 

          return status.report(__func__);
        }    

        TestOutcome scal(index_type N)
        {
          TestStatus status;

          ReSolve::LinAlgWorkspace* workspace = createLinAlgWorkspace(memspace_);
          ReSolve::VectorHandler handler(workspace);

          vector::Vector* x =  new vector::Vector(N);

          x->allocate(memspace_);

          x->setToConst(1.25, memspace_);

          real_type alpha = 3.5;

          //the answer is x[i] = 4.375;         
          handler.scal(&alpha, x, memspace_);
          status *= verifyAnswer(x, 4.375, memspace_);

          delete workspace;
          delete x;

          return status.report(__func__);
        }    

        TestOutcome massAxpy(index_type N, index_type K)
        {
          TestStatus status;

          ReSolve::LinAlgWorkspace* workspace = createLinAlgWorkspace(memspace_);
          ReSolve::VectorHandler handler(workspace);
          
          vector::Vector* x =  new vector::Vector(N, K);
          vector::Vector* y =  new vector::Vector(N);
          vector::Vector* alpha = new vector::Vector(K);;
          x->allocate(memspace_);
          y->allocate(memspace_);
          alpha->allocate(memspace_);

          y->setToConst(2.0, memspace_);
          alpha->setToConst(-1.0, memspace_);
          for (int ii = 0; ii < K; ++ii) {
            real_type c;
            if (ii % 2 == 0) {
              c = -1.0;
            } else {
              c = 0.5;
            }
            x->setToConst(ii, c, memspace_);
          }
          index_type r = K % 2;
          real_type res = (real_type) ((floor((real_type) K / 2.0) + r) * 1.0 + floor((real_type) K / 2.0) * (-0.5));

          handler.massAxpy(N, alpha, K, x, y, memspace_);
          status *= verifyAnswer(y, 2.0 - res, memspace_);


          delete workspace;
          delete x;
          delete y;
          delete alpha;

          return status.report(__func__);
        }    

        TestOutcome massDot(index_type N, index_type K)
        {
          TestStatus status;

          ReSolve::LinAlgWorkspace* workspace = createLinAlgWorkspace(memspace_);
          ReSolve::VectorHandler handler(workspace);
          
          vector::Vector* x =  new vector::Vector(N, K);
          vector::Vector* y =  new vector::Vector(N, 2);
          vector::Vector* res = new vector::Vector(K, 2);
          x->allocate(memspace_);
          y->allocate(memspace_);
          res->allocate(memspace_);
          
          x->setToConst(1.0, memspace_);
          y->setToConst(-1.0, memspace_);
          handler.massDot2Vec(N, x, K, y, res, memspace_);
          
          status *= verifyAnswer(res, (-1.0) * (real_type) N, memspace_);

          delete workspace;
          delete x;
          delete y;
          delete res;
          return status.report(__func__);
        }    

        TestOutcome gemv(index_type N,  index_type K)
        {
          TestStatus status;
          ReSolve::LinAlgWorkspace* workspace = createLinAlgWorkspace(memspace_);
          ReSolve::VectorHandler handler(workspace);
          vector::Vector* V = new vector::Vector(N, K);
          // for the test with NO TRANSPOSE
          vector::Vector* yN = new vector::Vector(K); 
          vector::Vector* xN = new vector::Vector(N);
          // for the test with TRANSPOSE
          vector::Vector* yT = new vector::Vector(N);
          vector::Vector* xT = new vector::Vector(K);
          
          V->allocate(memspace_);
          yN->allocate(memspace_);
          xN->allocate(memspace_);
          yT->allocate(memspace_);
          xT->allocate(memspace_);

          V->setToConst(1.0, memspace_);
          yN->setToConst(-1.0, memspace_);
          xN->setToConst(.5, memspace_);
          yT->setToConst(-1.0, memspace_);
          xT->setToConst(.5, memspace_);
          
          real_type alpha = -1.0;
          real_type beta = 1.0;
          handler.gemv("N", N, K, &alpha, &beta, V, yN, xN, memspace_);
          status *= verifyAnswer(xN, (real_type) (K) + 0.5, memspace_);
          handler.gemv("T", N, K, &alpha, &beta, V, yT, xT, memspace_);
          status *= verifyAnswer(xT, (real_type) (N) + 0.5, memspace_);

          return status.report(__func__);
        }    

      private:
        std::string memspace_{"cpu"};

        /** @brief Slaven's "factory" method - it would use correct constructor to create cuda (or other) workspace
        */ 
        LinAlgWorkspace* createLinAlgWorkspace(std::string memspace)
        {
          if (memspace == "cuda") {
            LinAlgWorkspaceCUDA* workspace = new LinAlgWorkspaceCUDA();
            workspace->initializeHandles();
            return workspace;
          } 
          // If not CUDA, return default
          return (new LinAlgWorkspace());
        }
        // we can verify through norm but that would defeat the purpose of testing vector handler...
        bool verifyAnswer(vector::Vector* x, real_type answer, std::string memspace)
        {
          bool status = true;
          if (memspace != "cpu") {
            x->copyData(memspace, "cpu");
          }

          for (index_type i = 0; i < x->getSize(); ++i) {
            // std::cout << x->getData("cpu")[i] << "\n";
            if (!isEqual(x->getData("cpu")[i], answer)) {
              status = false;
              std::cout << "Solution vector element x[" << i << "] = " << x->getData("cpu")[i]
                << ", expected: " << answer << "\n";
              break; 
            }
          }
          return status;
        }
    };//class
  }
}

