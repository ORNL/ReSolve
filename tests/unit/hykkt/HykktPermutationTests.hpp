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

        int perm[3] = {2, 0, 1};

        matrix::Csr hes(n, n, nnz_hes);
        matrix::Csr jac(m, n, nnz_jac);
        matrix::Csr jac_tr(n, m, nnz_jac);
        getTestData(&hes, &jac, &jac_tr);

        // correct results
        int hes_prc_i[4] = {0, 2, 4, 6};
        int hes_prc_j[6] = {1, 2, 0, 1, 0, 2};

        int jac_pc_i[3] = {0, 2, 4};
        int jac_pc_j[4] = {0, 1, 0, 2};

        int jac_tr_pr_i[4] = {0, 2, 3, 4};
        int jac_tr_pr_j[4] = {0, 1, 0, 1};

        int result_prc_i[4] = {};
        int result_prc_j[6] = {};
        int result_pr_i[4]  = {};
        int result_pr_j[4]  = {};
        int result_pc_j[4]  = {};

        ReSolve::hykkt::Permutation pc = ReSolve::hykkt::Permutation(n, m, nnz_hes, nnz_jac, memspace_);

        pc.addMatrixInfo(&hes, &jac, &jac_tr);

        pc.addCustomPerm(perm);
        pc.invertPerm();

        // Test RC permutation
        pc.vecMapRC(result_prc_i, result_prc_j);
        printf("Comparing RC permutation of H\n");
        bool flagrc = verifyResults(hes_prc_i, result_prc_i, n + 1);
        flagrc &= verifyResults(hes_prc_j, result_prc_j, nnz_hes);
        printf(!flagrc ? "RC permutation failed\n" : "RC permutation passed\n");

        // Test R permutation
        pc.vecMapR(result_pr_i, result_pr_j);
        printf("Comparing R permutation of J_tr\n");
        bool flagr = verifyResults(jac_tr_pr_i, result_pr_i, m + 1);
        flagr &= verifyResults(jac_tr_pr_j, result_pr_j, nnz_jac);
        printf(!flagr ? "R permutation failed\n" : "R permutation passed\n");

        // Test C permutation
        pc.vecMapC(result_pc_j);
        printf("Comparing C permutation of J\n");
        bool flagc = verifyResults(jac_pc_j, result_pc_j, nnz_jac);
        printf(!flagc ? "C permutation failed\n" : "C permutation passed\n");

        if (memspace_ == memory::DEVICE)
        {
          hes.syncData(memory::DEVICE);
          jac.syncData(memory::DEVICE);
          jac_tr.syncData(memory::DEVICE);
        }

        double hes_prc_v[6]   = {4, 5, 1, 0, 3, 2};
        double jac_pc_v[4]    = {1, 0, 3, 2};
        double jac_tr_pr_v[4] = {2, 3, 0, 1};

        // Test mapIndex on H
        double* result_prc_v = allocateArray(nnz_hes);
        pc.mapIndex(ReSolve::hykkt::PERM_HES_V, hes.getValues(memspace_), result_prc_v);
        double* h_result_prc_v = bringToHost(result_prc_v, nnz_hes);
        printf("Comparing mapped H nonzero values\n");
        bool flagrc_v = verifyResults(hes_prc_v, h_result_prc_v, nnz_hes);
        printf(!flagrc_v ? "Map Index failed on H\n" : "Map Index passed on H\n");

        // Test mapIndex on J
        double* result_pc_v = allocateArray(nnz_jac);
        pc.mapIndex(ReSolve::hykkt::PERM_JAC_V, jac.getValues(memspace_), result_pc_v);
        double* h_result_pc_v = bringToHost(result_pc_v, nnz_jac);
        printf("Comparing mapped J nonzero values\n");
        bool flagc_v = verifyResults(jac_pc_v, h_result_pc_v, nnz_jac);
        printf(!flagc_v ? "Map Index failed on J\n" : "Map Index passed on J\n");

        // Test mapIndex on J transpose
        double* result_pr_v = allocateArray(nnz_jac);
        pc.mapIndex(ReSolve::hykkt::PERM_JAC_TR_V, jac_tr.getValues(memspace_), result_pr_v);
        double* h_result_pr_v = bringToHost(result_pr_v, nnz_jac);
        printf("Comparing mapped J_TR nonzero values\n");
        bool flagr_v = verifyResults(jac_tr_pr_v, h_result_pr_v, nnz_jac);
        printf(!flagr_v ? "Map Index failed on J_TR\n" : "Map Index passed on J_TR\n");

        double  h_indices[3] = {0, 1, 2};
        double* indices;
        if (memspace_ == memory::HOST)
        {
          indices = h_indices; // already on host
        }
        else
        {
          indices = allocateArray(n);
          mem_.copyArrayHostToDevice(indices, h_indices, n);
        }

        double* result = allocateArray(n);

        pc.mapIndex(ReSolve::hykkt::PERM_V, indices, result);

        double* h_result = bringToHost(result, n);

        printf("Comparing mapped permutation\n");
        bool flag_perm = true;
        for (int i = 0; i < n; i++)
        {
          if (h_result[i] != (double) perm[i])
          {
            printf("Mismatch in index %d: %f != %f\n", i, h_result[i], (double) perm[i]);
            flag_perm = false;
          }
        }
        printf(!flag_perm ? "Map Index failed on perm\n" : "Map Index passed on perm\n");

        bool flag_rev_perm = true;
        pc.mapIndex(ReSolve::hykkt::REV_PERM_V, result, indices);
        if (memspace_ == memory::HOST)
        {
          h_result = indices;
        }
        else
        {
          mem_.copyArrayDeviceToHost(h_result, indices, n);
        }

        printf("Comparing mapped reverse permutation\n");
        for (int i = 0; i < n; i++)
        {
          if (h_result[i] != i)
          {
            printf("Mismatch in index %d: %f != %f\n", i, h_result[i], (double) i);
            flag_rev_perm = false;
          }
        }

        printf(!flag_rev_perm ? "Map Index failed on reverse perm\n" : "Map Index passed on reverse perm\n");

        // Free allocated memory
        freeArray(result_prc_v);
        freeArray(result_pc_v);
        freeArray(result_pr_v);
        freeArray(result);

        if (memspace_ != memory::HOST)
        {
          freeArray(indices);
          delete[] h_result_prc_v;
          delete[] h_result_pc_v;
          delete[] h_result_pr_v;
          delete[] h_result;
        }

        // Final Test Outcome
        return (flagrc && flagr && flagc && flagrc_v && flagr_v && flagc_v && flag_perm && flag_rev_perm) ? PASS : FAIL;
      }

    private:
      MemoryHandler                mem_;
      ReSolve::memory::MemorySpace memspace_;

      void getTestData(matrix::Csr* hes,
                       matrix::Csr* jac,
                       matrix::Csr* jac_tr)
      {
        int n       = 3;
        int m       = 2;
        int nnz_hes = 6;
        int nnz_jac = 4;

        int    hes_i[4] = {0, 2, 4, 6};
        int    hes_j[6] = {0, 2, 1, 2, 0, 1};
        double hes_v[6] = {0, 1, 2, 3, 4, 5};

        int    jac_i[3] = {0, 2, 4};
        int    jac_j[4] = {0, 2, 1, 2};
        double jac_v[4] = {0, 1, 2, 3};

        int    jac_tr_i[4] = {0, 1, 2, 4};
        int    jac_tr_j[4] = {0, 1, 0, 1};
        double jac_tr_v[4] = {0, 1, 2, 3};

        hes->copyDataFrom(hes_i, hes_j, hes_v, memory::HOST, memory::HOST);
        jac->copyDataFrom(jac_i, jac_j, jac_v, memory::HOST, memory::HOST);
        jac_tr->copyDataFrom(jac_tr_i, jac_tr_j, jac_tr_v, memory::HOST, memory::HOST);
      }

      bool verifyResults(const int* expected, const int* actual, int size)
      {
        for (int i = 0; i < size; ++i)
        {
          if (expected[i] != actual[i])
          {
            printf("Mismatch at index %d: expected %d, got %d\n", i, expected[i], actual[i]);
            return false;
          }
        }
        return true;
      }

      bool verifyResults(const double* expected, const double* actual, int size)
      {
        for (int i = 0; i < size; ++i)
        {
          if (expected[i] != actual[i])
          {
            printf("Mismatch at index %d: expected %f, got %f\n", i, expected[i], actual[i]);
            return false;
          }
        }
        return true;
      }

      double* allocateArray(int n)
      {
        if (memspace_ == memory::HOST)
        {
          return new double[n];
        }
        else
        {
          double* arr;
          mem_.allocateArrayOnDevice(&arr, n);
          return arr;
        }
      }

      double* bringToHost(double* arr, int n)
      {
        if (memspace_ == memory::HOST)
        {
          return arr; // already on host
        }
        else
        {
          double* h_arr = new double[n];
          mem_.copyArrayDeviceToHost(h_arr, arr, n);
          return h_arr;
        }
      }

      void freeArray(double* arr)
      {
        if (memspace_ == memory::HOST)
        {
          delete[] arr;
        }
        else
        {
          mem_.deleteOnDevice(arr);
        }
      }
    }; // class HykktPermutationTests
  } // namespace tests
} // namespace ReSolve
