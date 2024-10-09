#pragma once
#include <string>
#include <vector>
#include <iomanip>
#include <resolve/GramSchmidt.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/Vector.hpp>
#include <tests/unit/TestBase.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

namespace ReSolve
{ 
  namespace tests
  {
    const real_type var1 = 0.17;
    const real_type var2 = 2.0;

    class GramSchmidtTests : TestBase
    {
      public:       
        GramSchmidtTests(ReSolve::VectorHandler& handler) : handler_(handler) 
        {
          if (handler_.getIsCudaEnabled() || handler_.getIsHipEnabled())
            memspace_ = memory::DEVICE;
          else
            memspace_ = memory::HOST;
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
            case GramSchmidt::mgs_two_sync:
              testname += " (Modified Gram-Schmidt 2-Sync)";
              break;
            case GramSchmidt::mgs_pm:
              testname += " (Post-Modern Modified Gram-Schmidt)";
              break;
            case GramSchmidt::cgs1:
              testname += " (Classical Gram-Schmidt)";
              status.expectFailure();
              break;
            case GramSchmidt::cgs2:
              testname += " (Reorthogonalized Classical Gram-Schmidt)";
              break;
          }

          vector::Vector V(N, 3); // we will be using a space of 3 vectors
          real_type* H = new real_type[9]; // In this case, Hessenberg matrix is NOT 3 x 2 ???
          real_type* aux_data = nullptr; // needed for setup

          V.allocate(memspace_);
          if (memspace_ == memory::DEVICE) {
            V.allocate(memory::HOST);
          }

          ReSolve::GramSchmidt GS(&handler_, var);
          GS.setup(N, 3);
          
          //fill 2nd and 3rd vector with values
          aux_data = V.getVectorData(1, memory::HOST);
          for (int i = 0; i < N; ++i) {
            if ( i % 2 == 0) {         
              aux_data[i] = constants::ONE;
            } else {
              aux_data[i] = var1;
            }
          }
          aux_data = V.getVectorData(2, memory::HOST);
          for (int i = 0; i < N; ++i) {
            if ( i % 3 > 0) {         
              aux_data[i] = constants::ZERO;
            } else {
              aux_data[i] = var2;
            }
          }
          V.setDataUpdated(memory::HOST); 
          V.syncData(memory::HOST, memspace_);

          //set the first vector to all 1s, normalize 
          V.setToConst(0, 1.0, memspace_);
          real_type nrm = handler_.dot(&V, &V, memspace_);
          nrm = sqrt(nrm);
          nrm = 1.0 / nrm;
          handler_.scal(&nrm, &V, memspace_);

          GS.orthogonalize(N, &V, H, 0); 
          GS.orthogonalize(N, &V, H, 1); 
          status *= verifyAnswer(V, 3);

          delete [] H;
          
          return status.report(testname.c_str());
        }    

      private:
        ReSolve::VectorHandler& handler_;
        ReSolve::memory::MemorySpace memspace_;

        // x is a multivector containing K vectors 
        bool verifyAnswer(vector::Vector& x, index_type K)
        {
          vector::Vector a(x.getSize()); 
          vector::Vector b(x.getSize());

          real_type ip; 
          bool status = true;

          for (index_type i = 0; i < K; ++i) {
            for (index_type j = 0; j < K; ++j) {
              a.update(x.getVectorData(i, memspace_), memspace_, memory::HOST);
              b.update(x.getVectorData(j, memspace_), memspace_, memory::HOST);
              ip = handler_.dot(&a, &b, memory::HOST);
              if ( (i != j) && !isEqual(ip, 0.0)) {
                status = false;
                std::cout << "Vectors " << i << " and " << j << " are not orthogonal!"
                          << " Inner product computed: " << ip << ", expected: " << 0.0 << "\n";
                break; 
              }
              if ( (i == j) && !isEqual(sqrt(ip), 1.0)) {           
                status = false;
                std::cout << std::setprecision(16);
                std::cout << "Vector " << i << " has norm: " << sqrt(ip)
                          << " expected: " << 1.0 << "\n";
                break; 
              }
            }
          }

          return status;
        }
   }; // class
  } // namespace tests
} // namespace ReSolve
