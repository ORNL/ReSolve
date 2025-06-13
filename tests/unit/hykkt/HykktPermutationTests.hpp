/**
 * @file HykktPermutationTests.hpp
 * @author Shaked Regev (regevs@ornl.gov)
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief Implementation of tests for class hykkt::Permutation
 * 
 */
#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>

#include <tests/unit/TestBase.hpp>
#include <resolve/hykkt/Permutation.hpp>
#include <resolve/hykkt/PermutationHandler.hpp>

namespace ReSolve
{
  namespace tests
  {
    /**
     * @brief Tests for class hykkt::Permutation
     * 
     */
    class HykktPermutationTests : public TestBase
    {
    public:
      HykktPermutationTests(ReSolve::hykkt::PermutationHandler* permutationHandler): permutationHandler_(permutationHandler)
      {
        // Determine memory space based on the handler capabilities
          if (permutationHandler_->getIsCudaEnabled() || permutationHandler_->getIsHipEnabled()) {
            memspace_ = memory::DEVICE;
          } else {
            memspace_ = memory::HOST;
          }
      }
      virtual ~HykktPermutationTests() {}

      TestOutcome permutation()
      {
        int n = 4;
        int m = 4;
        int nnz = 9;
        int a_i[5]    = {0, 2, 5, 7, 9};
        int a_j[9]    = {0, 2, 0, 1, 2, 1, 2, 1, 3};
        int a_prc_i[5] = {0, 2, 4, 6, 9};
        int a_prc_j[9] = {0, 3, 0, 1, 2, 3, 0, 1, 3};
        int a_pr_i[5]  = {0, 2, 4, 6, 9};
        int a_pr_j[9]  = {1, 2, 0, 2, 1, 3, 0, 1, 2};
        int a_pc_i[5]  = {0, 2, 5, 7, 9};
        int a_pc_j[9]  = {0, 1, 0, 1, 3, 0, 3, 2, 3};
        int b_i[5] = {0}; // Initialize row pointer
        int b_j[9] = {0}; // Initialize column indices
        int perm[4] = {2, 0, 3, 1};

        bool flagrc = false;
        bool flagr = false;
        bool flagc = false;

        ReSolve::hykkt::Permutation pc = ReSolve::hykkt::Permutation(permutationHandler_, n, nnz, nnz);

        pc.addHInfo(a_i, a_j);
        pc.addJInfo(a_i, a_j, n, m);
        pc.addJtInfo(a_i, a_j);
        pc.addPerm(perm);
        pc.invertPerm(memspace_);

        // Test RC permutation
        pc.vecMapRC(b_i, b_j, memspace_);
        printf("Comparing RC permutation\n");
        for (int i = 0; i < n + 1; i++) // Loop over row pointers (n+1)
        {
          if (a_prc_i[i] != b_i[i])
          {
            printf("Mismatch in row pointer %d\n", i);
            flagrc = true;
          }
        }
        for (int j = 0; j < nnz; j++) // Compare column indices
        {
          if (a_prc_j[j] != b_j[j])
          {
            printf("Mismatch in column index %d\n", j);
            flagrc = true;
          }
        }
        printf(flagrc ? "RC permutation failed\n" : "RC permutation passed\n");

        // Test R permutation
        pc.vecMapR(b_i, b_j, memspace_);
        printf("Comparing R permutation\n");
        for (int i = 0; i < n + 1; i++)
        {
          if (a_pr_i[i] != b_i[i])
          {
            printf("Mismatch in row pointer %d\n", i);
            flagr = true;
          }
        }
        for (int j = 0; j < nnz; j++)
        {
          if (a_pr_j[j] != b_j[j])
          {
            printf("Mismatch in column index %d\n", j);
            flagr = true;
          }
        }
        printf(flagr ? "R permutation failed\n" : "R permutation passed\n");

        // Test C permutation
        pc.vecMapC(b_j, memspace_);
        printf("Comparing C permutation\n");
        for (int i = 0; i < n + 1; i++)
        {
          if (a_pc_i[i] != a_i[i]) // Row pointers should match
          {
            printf("Mismatch in row pointer %d\n", i);
            flagc = true;
          }
        }
        for (int j = 0; j < nnz; j++)
        {
          if (a_pc_j[j] != b_j[j])
          {
            printf("Mismatch in column index %d\n", j);
            flagc = true;
          }
        }
        printf(flagc ? "C permutation failed\n" : "C permutation passed\n");

        // Final Test Outcome
        return (!flagrc && !flagr && !flagc) ? PASS : FAIL;
      }

      private:
        ReSolve::hykkt::PermutationHandler* permutationHandler_;
        ReSolve::memory::MemorySpace memspace_;
    }; // class HykktPermutationTests
  } // namespace tests
} // namespace ReSolve