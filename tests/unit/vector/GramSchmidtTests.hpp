#pragma once
#include <string>
#include <vector>
#include <iomanip>
#include <resolve/GramSchmidt.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/Vector.hpp>
#include <tests/unit/TestBase.hpp>
#include <resolve/LinAlgWorkspace.hpp>

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
          // status.skipTest();

          GramSchmidt gs1;
          status *= (gs1.getVariant() == GramSchmidt::mgs);
          status *= (gs1.getL() == nullptr);
          status *= !gs1.isSetupComplete();

          VectorHandler vh;
          GramSchmidt gs2(&vh, GramSchmidt::mgs_pm);
          status *= (gs2.getVariant() == GramSchmidt::mgs_pm);
          status *= (gs1.getL() == nullptr);
          status *= !gs1.isSetupComplete();

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

          ReSolve::LinAlgWorkspace* workspace = createLinAlgWorkspace(memspace_);
          ReSolve::VectorHandler* handler = new ReSolve::VectorHandler(workspace);

          vector::Vector* V = new vector::Vector(N, 3); // we will be using a space of 3 vectors
          real_type* H = new real_type[6]; //in this case, Hessenberg matrix is 3 x 2
          real_type* aux_data; // needed for setup

          V->allocate(memspace_);
          if (memspace_ != "cpu") {
            V->allocate("cpu");
          }


          ReSolve::GramSchmidt* GS = new ReSolve::GramSchmidt(handler, var);
          GS->setup(N, 3);
          
          //fill 2nd and 3rd vector with values
          aux_data = V->getVectorData(1, "cpu");
          for (int i = 0; i < N; ++i) {
            if ( i % 2 == 0) {         
              aux_data[i] = constants::ONE;
            } else {
              aux_data[i] = var1;
            }
          }
          aux_data = V->getVectorData(2, "cpu");
          for (int i = 0; i < N; ++i) {
            if ( i % 3 > 0) {         
              aux_data[i] = constants::ZERO;
            } else {
              aux_data[i] = var2;
            }
          }
          V->setDataUpdated("cpu"); 
          V->copyData("cpu", memspace_);

          //set the first vector to all 1s, normalize 
          V->setToConst(0, 1.0, memspace_);
          real_type nrm = handler->dot(V, V, memspace_);
          nrm = sqrt(nrm);
          nrm = 1.0 / nrm;
          handler->scal(&nrm, V, memspace_);

          GS->orthogonalize(N, V, H, 0, memspace_ ); 
          GS->orthogonalize(N, V, H, 1, memspace_ ); 

          status *= verifyAnswer(V, 3, handler, memspace_);
          
          delete workspace;
          delete [] H;
          delete V; 
          delete GS;
          
          return status.report(testname.c_str());
        }    

      private:
        std::string memspace_{"cuda"};

        // x is a multivector containing K vectors 
        bool verifyAnswer(vector::Vector* x, index_type K,  ReSolve::VectorHandler* handler, std::string memspace)
        {
          vector::Vector* a = new vector::Vector(x->getSize()); 
          vector::Vector* b = new vector::Vector(x->getSize());

          real_type ip; 
          bool status = true;

          for (index_type i = 0; i < K; ++i) {
            for (index_type j = 0; j < K; ++j) {
              a->update(x->getVectorData(i, memspace), memspace, "cpu");
              b->update(x->getVectorData(j, memspace), memspace, "cpu");
              ip = handler->dot(a, b, "cpu");
              
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
