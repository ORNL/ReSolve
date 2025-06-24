/**
 * @file Permutation.cpp
 * @author Shaked Regev (regevs@ornl.gov)
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief Implementation of the Permutation class.
 *
 *
 */
#include <cstdio>

#include "amd.h"
#include <resolve/hykkt/permutation/Permutation.hpp>

namespace ReSolve
{
  namespace hykkt
  {
    /**
     * @brief Permutation constructor
     *
     * @param[in] permutationHandler - Pointer to the PermutationHandler
     * @param[in] n_hes - Number of rows/columns in matrix H
     * @param[in] nnz_hes - Number of nonzeros in matrix H
     * @param[in] nnz_jac - Number of nonzeros in matrix J
     *
     * @post Member variables initialized, workspace allocated
     */
    Permutation::Permutation(int n_hes, int n_jac, int nnz_hes, int nnz_jac, memory::MemorySpace memspace)
      : n_hes_(n_hes),
        n_jac_(n_jac),
        m_jac_(n_hes),
        nnz_hes_(nnz_hes),
        nnz_jac_(nnz_jac),
        memspace_(memspace)
    {
      allocateWorkspace();
      perm_is_default_ = true;
    }

    /// Permutation destructor
    Permutation::~Permutation()
    {
      deleteWorkspace();
    }

    /**
     * @brief loads CSR structure for matrix H
     *
     * @param[in] hes_i - Row offsets for H
     * @param[in] hes_j - Column indices for H
     *
     * @pre Matrix data stored in the same memory space as was passed
     *      into the constructor
     *
     * @post hes_i_ set to hes_i, hes_j_ set to hes_j
     */
    void Permutation::addHInfo(matrix::Csr* hes)
    {
      hes_i_ = hes->getRowData(memspace_);
      hes_j_ = hes->getColData(memspace_);
    }

    /**
     * @brief loads CSR structure for matrix J
     *
     * @param[in] jac_i - Row offsets for J
     * @param[in] jac_j - Column indices for j
     *
     * @pre Matrix data stored in the same memory space as was passed
     *      into the constructor
     *
     * @post jac_i_ set to jac_i, jac_j_ set to jac_j, n_jac_ set to n_jac,
     * m_jac_ set to m_jac
     */
    void Permutation::addJInfo(matrix::Csr* jac)
    {
      jac_i_ = jac->getRowData(memspace_);
      jac_j_ = jac->getColData(memspace_);
    }

    /**
     * @brief loads CSR structure for matrix Jt
     *
     * @param[in] jac_tr_i - Row offsets for Jt
     * @param[in] jac_tr_j - Column indices for Jt
     *
     * @pre Matrix data stored in the same memory space as was passed
     *      into the constructor
     *
     * @pre
     * @post jac_tr_i_ set to jac_tr_i, jac_tr_j_ set to jac_tr_j
     */
    void Permutation::addJtInfo(matrix::Csr* jac_tr)
    {
      jac_tr_i_ = jac_tr->getRowData(memspace_);
      jac_tr_j_ = jac_tr->getColData(memspace_);
    }

    /**
     * @brief sets custom permutation of matrix
     *
     * @param[in] perm - permutation vector stored in the same memory space as the matrix data
     *
     * @post perm points to custom_perm out of scope so perm_is_default
     *       set to false so that custom_perm not deleted twice in destructors,
     *       permutation vector copied onto device d_perm
     */
    void Permutation::addCustomPerm(int* custom_perm)
    {
      if (perm_is_default_)
      {
        if (memspace_ == memory::HOST)
        {
          delete[] perm_;
        }
        else
        {
          mem_.deleteOnDevice(perm_);
        }
      }
      perm_is_default_ = false;
      perm_            = custom_perm;
    }

    /**
     * @brief Uses Symmetric Approximate Minimum Degree
     *        to reduce zero-fill in Cholesky Factorization
     *
     * @pre Member variables n_hes_, nnz_hes_, hes_i_, hes_j_ have been
     *      initialized to the dimensions of matrix H, the number
     *      of nonzeros it has, its row offsets, and column arrays
     *
     * @post perm is the permutation vector that implements symAmd
     *       on the 2x2 system
     */
    void Permutation::symAmd()
    {
      double Control[AMD_CONTROL], Info[AMD_INFO];

      amd_defaults(Control);
      amd_control(Control);

      int result = amd_order(n_hes_, hes_i_, hes_j_, perm_, Control, Info);

      if (result != AMD_OK)
      {
        printf("AMD failed\n");
        exit(1);
      }
    }

    /**
     * @brief Creates reverse permutation of perm and copies onto device
     *
     * @pre Member variables n_hes_, perm intialized to dimension of matrix
     *      and to a permutation vector
     *
     * @post rev_perm is now the reverse permuation of perm and copied onto
     *       the device d_perm
     */
    void Permutation::invertPerm()
    {
      permutationHandler_.reversePerm(n_hes_, perm_, rev_perm_, memspace_);
    }

    /**
     * @brief Creates permutation of rows and columns of matrix
     * and copies onto device
     *
     * @param[out] perm_i - row offsets of permutation
     * @param[out] perm_j - column indices of permutation
     *
     * @pre Member variables n_hes_, nnz_hes_, hes_i_, hes_j_, perm, rev_perm
     *      initialized to the dimension of matrix H, number of nonzeros
     *      in H, row offsets for H, column indices for H, permutation
     *      and reverse permutation of H
     *
     * @post perm_map_h is now permuted rows/columns of H and copied onto
     *       the device d_perm_map_h
     */
    void Permutation::vecMapRC(int* perm_i, int* perm_j)
    {
      permutationHandler_.makeVecMapRC(n_hes_, hes_i_, hes_j_, perm_, rev_perm_, perm_i, perm_j, perm_map_hes_, memspace_);
    }

    /**
     * @brief Creates the permutation of the columns of matrix J
     * and copies onto device
     *
     * @param[out] perm_j - column indices of permutation
     *
     * @pre Member variables n_jac_, nnz_jac_, jac_i_, jac_j_, rev_perm initialized
     *      to the dimension of matrix J, number of nonzeros in J, row
     *      offsets for J, column indices for J, and reverse permutation
     *
     * @post perm_map_jac is now the column permutation and is copied onto
     *       the device d_perm_map_jac
     */
    void Permutation::vecMapC(int* perm_j)
    {
      permutationHandler_.makeVecMapC(n_jac_, jac_i_, jac_j_, rev_perm_, perm_j, perm_map_jac_, memspace_);
    }

    /**
     * @brief Creates the permutation of the rows of matrix Jt
     * and copies onto device
     *
     * @param[out] perm_i - row offsets of permutation
     * @param[out] perm_j - column indices of permutation
     *
     * @pre Member variables m_jac_, nnz_jac_, jac_tr_i_, jac_tr_j_, initialized to
     *      the dimension of matrix J, the number of nonzeros in J, the
     *      row offsets for J transpose, the column indices for J transpose
     *
     * @post perm_map_jac_tr is now the permuations of the rows of J transpose
     *       and is copied onto the device d_perm_map_jac_tr
     */
    void Permutation::vecMapR(int* perm_i, int* perm_j)
    {
      permutationHandler_.makeVecMapR(m_jac_, jac_tr_i_, jac_tr_j_, perm_, perm_i, perm_j, perm_map_jac_tr_, memspace_);
    }

    /**
     * @brief maps the permutated values of old_val to new_val
     *
     * @param[in] permutation - the type of permutation of the 2x2 system
     * @param[in] old_val     - the old values in the matrix
     * @param[out] new_val     - the permuted values
     *
     * @pre Member variables n_hes_, nnz_hes_, nnz_jac_, d_perm, d_rev_perm,
     *      d_perm_map_h, d_perm_map_jac, d_perm_map_jac_tr initialized to
     *      the dimension of matrix H, number of nonzeros in H, number
     *      of nonzeros in matrix J, the device permutation and reverse
     *      permutation vectors, the device permutation mappings for
     *      H, J, and J transpose
     *
     * @post new_val contains the permuted old_val
     */
    void Permutation::mapIndex(PermutationType permutation,
                               double*         old_val,
                               double*         new_val)
    {
      switch (permutation)
      {
      case PERM_V:
        permutationHandler_.mapIdx(n_hes_, perm_, old_val, new_val, memspace_);
        break;
      case REV_PERM_V:
        permutationHandler_.mapIdx(n_hes_, rev_perm_, old_val, new_val, memspace_);
        break;
      case PERM_HES_V:
        permutationHandler_.mapIdx(nnz_hes_, perm_map_hes_, old_val, new_val, memspace_);
        break;
      case PERM_JAC_V:
        permutationHandler_.mapIdx(nnz_jac_, perm_map_jac_, old_val, new_val, memspace_);
        break;
      case PERM_JAC_TR_V:
        permutationHandler_.mapIdx(nnz_jac_, perm_map_jac_tr_, old_val, new_val, memspace_);
        break;
      default:
        printf("Valid arguments are PERM_V, REV_PERM_V, PERM_H_V, PERM_J_V, PERM_JT_V\n");
      }
    }

    /**
     * @brief deletes memory allocated for permutation vectors
     *
     * @pre perm_, rev_perm_, perm_map_h, perm_map_jac, perm_map_jac_tr
     *  are allocated memory
     *
     * @post memory allocated for perm_, rev_perm_, perm_map_h, perm_map_jac,
     *      perm_map_jac_tr is deleted
     */
    void Permutation::deleteWorkspace()
    {
      if (memspace_ == memory::HOST)
      {
        if (perm_is_default_)
        {
          delete[] perm_;
        }
        delete[] rev_perm_;
        delete[] perm_map_hes_;
        delete[] perm_map_jac_;
        delete[] perm_map_jac_tr_;
      }
      else
      {
        if (perm_is_default_)
        {
          mem_.deleteOnDevice(perm_);
        }
        mem_.deleteOnDevice(rev_perm_);
        mem_.deleteOnDevice(perm_map_hes_);
        mem_.deleteOnDevice(perm_map_jac_);
        mem_.deleteOnDevice(perm_map_jac_tr_);
      }
    }

    /**
     * @brief allocates memory on host for permutation vectors
     *
     * @pre Member variables n_hes_, nnz_hes_, nnz_jac_ are initialized to the
     *      dimension of matrix H, number of nonzeros in H, and number of
     *      nonzeros in matrix J
     *
     * @post perm_ and rev_perm_ are now vectors with size n_hes_, perm_map_h
     *       is now a vector with size nnz_hes_, perm_map_jac and perm_map_jac_tr
     *       are now vectors with size nnz_jac_
     */
    void Permutation::allocateWorkspace()
    {
      if (memspace_ == memory::HOST)
      {
        perm_            = new int[n_hes_];
        rev_perm_        = new int[n_hes_];
        perm_map_hes_    = new int[nnz_hes_];
        perm_map_jac_    = new int[nnz_jac_];
        perm_map_jac_tr_ = new int[nnz_jac_];
      }
      else
      {
        mem_.allocateArrayOnDevice(&perm_, n_hes_);
        mem_.allocateArrayOnDevice(&rev_perm_, n_hes_);
        mem_.allocateArrayOnDevice(&perm_map_hes_, nnz_hes_);
        mem_.allocateArrayOnDevice(&perm_map_jac_, nnz_jac_);
        mem_.allocateArrayOnDevice(&perm_map_jac_tr_, nnz_jac_);
      }
    }
  } // namespace hykkt
} //  namespace ReSolve
