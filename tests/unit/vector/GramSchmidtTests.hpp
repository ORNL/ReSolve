#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <resolve/GramSchmidt.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/Vector.hpp>
#include <tests/unit/TestBase.hpp>

namespace ReSolve { 
  namespace tests {
    class GramSchmidtTests : TestBase
    {
      public:       
        GramSchmidtTests(std::string memspace) : memspace_(memspace) 
        {
        }

        virtual ~GramSchmidtTests()
        {
        }

        TestOutcome GramSchmidtConstructor()
        {
          TestStatus status;
          status.skipTest();

          return status.report(__func__);
        }
 
        TestOutcome orthogonalize(index_type N, GSVariant var)
        {
          TestStatus status;

          ReSolve::LinAlgWorkspace* workspace = createLinAlgWorkspace(memspace_);
          ReSolve::VectorHandler* handler = new ReSolve::VectorHandler(workspace);

          vector::Vector* V = new vector::Vector(N, 3); // we will be using a space of 3 vectors
          real_type* H = new real_type[6]; //in this case, Hessenberg matrix is 3 x 2
          real_type* aux_data; // needed for setup

          V->allocate(memspace_);
          if (memspace_ != "cpu") {          
            V->allocate("cpu");
          }

         //set the first vector to all 1s, normalize 
          V->setToConst(0, 1.0, memspace_);
          real_type nrm = handler->dot(V, V, memspace_);
          nrm = sqrt(nrm);
          nrm = 1.0 / nrm;
          handler->scal(&nrm, V, memspace_);
                 
          ReSolve::GramSchmidt* GS = new ReSolve::GramSchmidt(handler, var);
          GS->setup(N, 3);
                    
          aux_data = V->getVectorData(1, "cpu");
          for (int i = 0; i < N; ++i) {
            if ( i % 2 == 0) {         
              aux_data[i] = 1.0;
            } else {
              aux_data[i] = 0.17;
            }
          }
          aux_data = V->getVectorData(2, "cpu");
          for (int i = 0; i < N; ++i) {
            if ( i % 3 > 0) {         
              aux_data[i] = 0.0;
            } else {
              aux_data[i] = 2.0;
            }
          }

          V->setDataUpdated("cpu"); 
          V->copyData("cpu", memspace_);
         
          GS->orthogonalize(N, V, H, 0, memspace_ ); 
          GS->orthogonalize(N, V, H, 1, memspace_ ); 
          
          status *= verifyAnswer(V, 3, handler, memspace_);
          
          delete workspace;
          delete [] H;
          delete V; 
          delete GS;
          return status.report(__func__);
        }    
private:
        std::string memspace_{"cuda"};

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
        
        // x is a multivector containing K vectors 
        bool verifyAnswer(vector::Vector* x, index_type K,  ReSolve::VectorHandler* handler, std::string memspace)
        {
          vector::Vector* a = new vector::Vector(x->getSize()); 
          vector::Vector* b = new vector::Vector(x->getSize());
          
          real_type ip; 
          bool status = true;
          
          for (index_type i = 0; i < K; ++i) {
            for (index_type j = 0; j < K; ++j) {

              a->setData(x->getVectorData(i, memspace), memspace);
              b->setData(x->getVectorData(j, memspace), memspace);
              ip = handler->dot(a, b, memspace_);
                
              if ( (i != j) && (abs(ip) > 1e-14)) {
                status = false;
                std::cout << "Vectors" << i << " and " << j << "are not orthogonal, inner product: " << ip << " expected: "<< 0.0 <<"\n";
                break; 
              }
              if ( (i == j) && (abs(sqrt(ip)) != 1.0)) {
                status = false;
                std::cout << "Vector" << i << " has norm" << sqrt(ip) << " expected: "<< 1.0 <<"\n";
                break; 
              }
            }
          }
          return status;
        }
    }; // class
  }
}
