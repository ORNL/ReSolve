#pragma once
#include <string>
#include <vector>
#include <iomanip>
#include <resolve/GramSchmidt.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/Vector.hpp>
#include <tests/unit/TestBase.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

namespace ReSolve { 
  namespace tests {
    const real_type var1 = 0.17;
    const real_type var2 = 2.0;
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

          VectorHandler vh;
          GramSchmidt gs2(&vh, GramSchmidt::mgs_pm);
          status *= (gs2.getVariant() == GramSchmidt::mgs_pm);

          return status.report(__func__);
        }

        TestOutcome orthogonalize(index_type N, GramSchmidt::GSVariant var)
        {
          TestStatus status;

          std::string testname(__func__);
          switch(var)
          {
            case GramSchmidt::mgs:
              testname += " (Modified Gram-Schmidt)";
              break;
            case GramSchmidt::mgs_two_synch:
              testname += " (Modified Gram-Schmidt 2-Sync)";
              break;
            case GramSchmidt::mgs_pm:
              testname += " (Post-Modern Modified Gram-Schmidt)";
              break;
            case GramSchmidt::cgs1:
              testname += " (Classical Gram-Schmidt)";
              break;
            case GramSchmidt::cgs2:
              testname += " (Reorthogonalized Classical Gram-Schmidt)";
              break;
          }

          ReSolve::memory::MemorySpace ms;
          if (memspace_ == "cpu")
            ms = memory::HOST;
          else
            ms = memory::DEVICE;

          ReSolve::VectorHandler* handler = createVectorHandler();

          vector::Vector* V = new vector::Vector(N, 3); // we will be using a space of 3 vectors
          real_type* H = new real_type[6]; //in this case, Hessenberg matrix is 3 x 2
          real_type* aux_data; // needed for setup

          V->allocate(ms);
          if (ms != memory::HOST) {
            V->allocate(memory::HOST);
          }


          ReSolve::GramSchmidt* GS = new ReSolve::GramSchmidt(handler, var);
          GS->setup(N, 3);
          
          //fill 2nd and 3rd vector with values
          aux_data = V->getVectorData(1, memory::HOST);
          for (int i = 0; i < N; ++i) {
            if ( i % 2 == 0) {         
              aux_data[i] = constants::ONE;
            } else {
              aux_data[i] = var1;
            }
          }
          aux_data = V->getVectorData(2, memory::HOST);
          for (int i = 0; i < N; ++i) {
            if ( i % 3 > 0) {         
              aux_data[i] = constants::ZERO;
            } else {
              aux_data[i] = var2;
            }
          }
          V->setDataUpdated(memory::HOST); 
          V->copyData(memory::HOST, ms);

          //set the first vector to all 1s, normalize 
          V->setToConst(0, 1.0, ms);
          real_type nrm = handler->dot(V, V, ms);
          nrm = sqrt(nrm);
          nrm = 1.0 / nrm;
          handler->scal(&nrm, V, ms);

          GS->orthogonalize(N, V, H, 0); 
          GS->orthogonalize(N, V, H, 1); 

          status *= verifyAnswer(V, 3, handler, memspace_);
          
          delete handler;
          delete [] H;
          delete V; 
          delete GS;
          
          return status.report(testname.c_str());
        }    

      private:
        std::string memspace_{"cuda"};

        ReSolve::VectorHandler* createVectorHandler()
        {
          if (memspace_ == "cpu") { // TODO: Fix memory leak here
            LinAlgWorkspaceCpu* workpsace = new LinAlgWorkspaceCpu();
            return new VectorHandler(workpsace);
#ifdef RESOLVE_USE_CUDA
          } else if (memspace_ == "cuda") {
            LinAlgWorkspaceCUDA* workspace = new LinAlgWorkspaceCUDA();
            workspace->initializeHandles();
            return new VectorHandler(workspace);
#endif
          } else {
            std::cout << "ReSolve not built with support for memory space " << memspace_ << "\n";
          }
          return nullptr;
        }

        // x is a multivector containing K vectors 
        bool verifyAnswer(vector::Vector* x, index_type K,  ReSolve::VectorHandler* handler, std::string memspace)
        {
          ReSolve::memory::MemorySpace ms;
          if (memspace == "cpu")
            ms = memory::HOST;
          else
            ms = memory::DEVICE;

          vector::Vector* a = new vector::Vector(x->getSize()); 
          vector::Vector* b = new vector::Vector(x->getSize());

          real_type ip; 
          bool status = true;

          for (index_type i = 0; i < K; ++i) {
            for (index_type j = 0; j < K; ++j) {
              a->update(x->getVectorData(i, ms), ms, memory::HOST);
              b->update(x->getVectorData(j, ms), ms, memory::HOST);
              ip = handler->dot(a, b, memory::HOST);
              if ( (i != j) && (abs(ip) > 1e-14)) {
                status = false;
                std::cout << "Vectors " << i << " and " << j << " are not orthogonal!"
                          << " Inner product computed: " << ip << ", expected: " << 0.0 << "\n";
                break; 
              }
              if ( (i == j) && !isEqual(abs(sqrt(ip)), 1.0)) {
                status = false;
                std::cout << std::setprecision(16);
                std::cout << "Vector " << i << " has norm: " << sqrt(ip) << " expected: "<< 1.0 <<"\n";
                break; 
              }
            }
          }
          delete a;
          delete b;
          return status;
        }
    }; // class
  }
}
