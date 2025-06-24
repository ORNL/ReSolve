/**
 * @file HykktPermutationTests.hpp
 * @author Shaked Regev (regevs@ornl.gov)
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief Implementation of tests for class hykkt::Permutation
 *
 */
#pragma once

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <resolve/hykkt/permutation/Permutation.hpp>
#include <resolve/hykkt/permutation/PermutationHandler.hpp>
#include <tests/unit/TestBase.hpp>

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
      HykktPermutationTests(memory::MemorySpace memspace = memory::HOST)
      {
        memspace_ = memspace;
      }

      virtual ~HykktPermutationTests()
      {
      }

      TestOutcome permutationTest()
      {
        // n_hes = m_jac = 3, n_jac = 2
        int n       = 3;
        int m       = 2;
        int nnz_hes = 6;
        int nnz_jac = 4;

        int perm[4] = {2, 0, 1};

        int hes_i[4] = {0, 2, 4, 6};
        int hes_j[6] = {0, 2, 1, 2, 0, 1};

        int hes_prc_i[4] = {0, 2, 4, 6};
        int hes_prc_j[6] = {1, 2, 0, 1, 0, 2};

        int jac_i[3] = {0, 2, 4};
        int jac_j[4] = {0, 2, 1, 2};

        int jac_pc_i[3] = {0, 2, 4};
        int jac_pc_j[4] = {0, 1, 0, 2};

        int jac_tr_i[4] = {0, 1, 2, 4};
        int jac_tr_j[4] = {0, 1, 0, 1};

        int jac_tr_pr_i[4] = {0, 2, 3, 4};
        int jac_tr_pr_j[4] = {0, 1, 0, 1};

        int result_prc_i[4] = {};
        int result_prc_j[6] = {};
        int result_pr_i[4]  = {};
        int result_pr_j[4]  = {};
        int result_pc_j[4]  = {};

        bool flagrc = false;
        bool flagc  = false;
        bool flagr  = false;

        ReSolve::hykkt::Permutation pc = ReSolve::hykkt::Permutation(n, nnz_hes, nnz_jac, memspace_);

        pc.addHInfo(hes_i, hes_j);
        pc.addJInfo(jac_i, jac_j, m, n);
        pc.addJtInfo(jac_tr_i, jac_tr_j);

        pc.addPerm(perm);
        pc.invertPerm();

        // Test RC permutation
        pc.vecMapRC(result_prc_i, result_prc_j);
        printf("Comparing RC permutation of H\n");
        for (int i = 0; i < n + 1; i++) // Loop over row pointers (n+1)
        {
          if (hes_prc_i[i] != result_prc_i[i])
          {
            printf("Mismatch in row pointer %d\n", i);
            flagrc = true;
          }
        }
        for (int j = 0; j < nnz_hes; j++) // Compare column indices
        {
          if (hes_prc_j[j] != result_prc_j[j])
          {
            printf("Mismatch in column index %d\n", j);
            flagrc = true;
          }
        }
        printf(flagrc ? "RC permutation failed\n" : "RC permutation passed\n");

        // Test R permutation
        pc.vecMapR(result_pr_i, result_pr_j);
        printf("Comparing R permutation of J_tr\n");
        for (int i = 0; i < n + 1; i++)
        {
          if (jac_tr_pr_i[i] != result_pr_i[i])
          {
            printf("Mismatch in row pointer %d: %d != %d\n", i, jac_tr_pr_i[i], result_pr_i[i]);
            flagr = true;
          }
        }
        for (int j = 0; j < nnz_jac; j++)
        {
          if (jac_tr_pr_j[j] != result_pr_j[j])
          {
            printf("Mismatch in column index %d: %d != %d\n", j, jac_tr_pr_j[j], result_pr_j[j]);
            flagr = true;
          }
        }
        printf(flagr ? "R permutation failed\n" : "R permutation passed\n");

        // Test C permutation
        pc.vecMapC(result_pc_j);
        printf("Comparing C permutation of J\n");
        for (int j = 0; j < nnz_jac; j++)
        {
          if (jac_pc_j[j] != result_pc_j[j])
          {
            printf("Mismatch in column index %d\n", j);
            flagc = true;
          }
        }
        printf(flagc ? "C permutation failed\n" : "C permutation passed\n");

        double hes_v[6] = {0, 1, 2, 3, 4, 5};
        double hes_prc_v[6] = {4, 5, 1, 0, 3, 2};

        double jac_v[4] = {0, 1, 2, 3};
        double jac_pc_v[4] = {1, 0, 3, 2};

        double jac_tr_v[4] = {0, 1, 2, 3};
        double jac_tr_pr_v[4] = {2, 3, 0, 1};

        double result_prc_v[6] = {};
        double result_pr_v[4]  = {};
        double result_pc_v[4]  = {};

        bool flagrc_v = false;
        bool flagr_v  = false;
        bool flagc_v  = false;

        pc.mapIndex(ReSolve::hykkt::PERM_HES_V, hes_v, result_prc_v);
        printf("Comparing mapped H nonzero values\n");
        for (int j = 0; j < nnz_hes; j++)
        {
          if (hes_prc_v[j] != result_prc_v[j])
          {
            printf("Mismatch in column index %d\n", j);
            flagrc_v = true;
          }
        }
        printf(flagrc_v ? "Map Index failed on H\n" : "Map Index passed on H\n");

        pc.mapIndex(ReSolve::hykkt::PERM_JAC_V, jac_v, result_pc_v);
        printf("Comparing mapped J nonzero values\n");
        for (int j = 0; j < nnz_jac; j++)
        {
          if (jac_pc_v[j] != result_pc_v[j])
          {
            printf("Mismatch in column index %d\n", j);
            flagc_v = true;
          }
        }
        printf(flagc_v ? "Map Index failed on J\n" : "Map Index passed on J\n");

        pc.mapIndex(ReSolve::hykkt::PERM_JAC_V, jac_v, result_pc_v);
        printf("Comparing mapped J nonzero values\n");
        for (int j = 0; j < nnz_jac; j++)
        {
          if (jac_pc_v[j] != result_pc_v[j])
          {
            printf("Mismatch in column index %d\n", j);
            flagr_v = true;
          }
        }
        printf(flagr_v ? "Map Index failed on J_TR\n" : "Map Index passed on J_TR\n");

        double indices[3] = {0, 1, 2};
        double result[3] = {};
        bool flag_perm = false;
        bool flag_rev_perm = false;
        
        pc.mapIndex(ReSolve::hykkt::PERM_V, indices, result);
        printf("Comparing mapped permutation\n");
        for (int i = 0; i < n; i++)
        {
          if (result[i] != perm[i])
          {
            printf("Mismatch in index %d: %f != %f\n", i, result[i], (double) perm[i]);
            flag_perm = true;
          }
        }

        printf(flag_perm ? "Map Index failed on perm\n" : "Map Index passed on perm\n");  

        pc.mapIndex(ReSolve::hykkt::REV_PERM_V, result, indices);
        printf("Comparing mapped reverse permutation\n");
        for (int i = 0; i < n; i++)
        {
          if (indices[i] != i)
          {
            printf("Mismatch in index %d: %f != %f\n", i, indices[i], (double) i);
            flag_rev_perm = true;
          }
        }

        printf(flag_rev_perm ? "Map Index failed on reverse perm\n" : "Map Index passed on reverse perm\n");

        // Final Test Outcome
        return (!flagrc && !flagr && !flagc && !flagrc_v && 
          !flagr_v && !flagc_v && !flag_perm && !flag_rev_perm) ? PASS : FAIL;
      }

    private:
      ReSolve::memory::MemorySpace memspace_;
    }; // class HykktPermutationTests
  } // namespace tests
} // namespace ReSolve
