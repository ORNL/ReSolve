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
  using out = ReSolve::io::Logger;

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
    Permutation::Permutation(index_type n_hes, index_type n_jac, index_type nnz_hes, index_type nnz_jac, memory::MemorySpace memspace)
      : n_hes_(n_hes),
        n_jac_(n_jac),
        m_jac_(n_hes),
        nnz_hes_(nnz_hes),
        nnz_jac_(nnz_jac),
        memspace_(memspace)
    {
      allocateWorkspace();
      perm_is_default_ = true;

      cpuImpl_ = new CpuPermutationKernels();
#ifdef RESOLVE_USE_CUDA
      devImpl_ = new CudaPermutationKernels();
#elif defined(RESOLVE_USE_HIP)
      devImpl_ = new HipPermutationKernels();
#else
      if (memspace_ == memory::DEVICE)
      {
        out::error() << "No GPU support enabled, and memory space set to DEVICE.\n";
        exit(1);
      }
#endif
    }

    /// Permutation destructor
    Permutation::~Permutation()
    {
      delete cpuImpl_;
      if (devImpl_ != nullptr)
      {
        delete devImpl_;
      }
      deleteWorkspace();
    }

    /**
     * @brief loads CSR structure for matrix H
     *
     * @param hes - pointer to matrix H
     * @param jac - pointer to matrix J
     * @param jac_tr - pointer to matrix J transpose
     *
     * @pre If this is the first time this function is called, then the data
     *      is stored on HOST, otherwise, the data is stored in the same memory
     *      space as passed to the constructor.
     */
    void Permutation::addMatrixInfo(matrix::Csr* hes, matrix::Csr* jac, matrix::Csr* jac_tr)
    {
      hes_i_    = hes->getRowData(memory::HOST);
      hes_j_    = hes->getColData(memory::HOST);
      jac_i_    = jac->getRowData(memory::HOST);
      jac_j_    = jac->getColData(memory::HOST);
      jac_tr_i_ = jac_tr->getRowData(memory::HOST);
      jac_tr_j_ = jac_tr->getColData(memory::HOST);
    }

    /**
     * @brief Set a custom permutation
     *
     * This function allows the user to set a custom permutation vector, rather
     * than use symmetric approximate minimum degree, the function symAmd(), to generate one.
     *
     * @param[in] custom_perm - custom permutation vector stored on HOST
     *
     * @post perm_ points to the custom permutation vector, and is not owned by the Permutation class anymore.
     */
    void Permutation::addCustomPerm(index_type* custom_perm)
    {
      if (perm_is_default_)
      {
        delete[] perm_;
      }
      perm_is_default_ = false;
      perm_            = custom_perm;

      if (memspace_ == memory::DEVICE)
      {
        mem_.copyArrayHostToDevice(d_perm_, perm_, n_hes_);
      }
    }

    /**
     * @brief Uses Symmetric Approximate Minimum Degree
     *        to reduce zero-fill in Cholesky Factorization
     *
     * @pre addMatrixInfo() has been called
     *
     * @post perm_ is the permutation vector that implements symAmd
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
        out::error() << "AMD failed\n";
        exit(1);
      }

      if (memspace_ == memory::DEVICE)
      {
        mem_.copyArrayHostToDevice(d_perm_, perm_, n_hes_);
      }
    }

    /**
     * @brief Creates reverse permutation of perm and copies onto device
     *
     * @pre Either addCustomPerm() or symAmd() has been called
     *
     * @post rev_perm is now the reverse permuation of perm and copied onto
     *       the device d_perm
     */
    void Permutation::invertPerm()
    {
      cpuImpl_->reversePerm(n_hes_, perm_, rev_perm_);
      if (memspace_ == memory::DEVICE)
      {
        mem_.copyArrayHostToDevice(d_rev_perm_, rev_perm_, n_hes_);
      }
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
    void Permutation::vecMapRC(index_type* perm_i, index_type* perm_j)
    {

      cpuImpl_->makeVecMapRC(n_hes_, hes_i_, hes_j_, perm_, rev_perm_, perm_i, perm_j, perm_map_hes_);
      if (memspace_ == memory::DEVICE)
      {
        mem_.copyArrayHostToDevice(d_perm_map_hes_, perm_map_hes_, nnz_hes_);
      }
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
    void Permutation::vecMapC(index_type* perm_j)
    {
      cpuImpl_->makeVecMapC(n_jac_, jac_i_, jac_j_, rev_perm_, perm_j, perm_map_jac_);
      if (memspace_ == memory::DEVICE)
      {
        mem_.copyArrayHostToDevice(d_perm_map_jac_, perm_map_jac_, nnz_jac_);
      }
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
    void Permutation::vecMapR(index_type* perm_i, index_type* perm_j)
    {
      cpuImpl_->makeVecMapR(m_jac_, jac_tr_i_, jac_tr_j_, perm_, perm_i, perm_j, perm_map_jac_tr_);
      if (memspace_ == memory::DEVICE)
      {
        mem_.copyArrayHostToDevice(d_perm_map_jac_tr_, perm_map_jac_tr_, nnz_jac_);
      }
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
                               real_type*      old_val,
                               real_type*      new_val)
    {
      index_type  length;
      index_type* apply_perm_;
      switch (permutation)
      {
      case PERM_V:
        length      = n_hes_;
        apply_perm_ = memspace_ == memory::HOST ? perm_ : d_perm_;
        break;
      case REV_PERM_V:
        length      = n_hes_;
        apply_perm_ = memspace_ == memory::HOST ? rev_perm_ : d_rev_perm_;
        break;
      case PERM_HES_V:
        length      = nnz_hes_;
        apply_perm_ = memspace_ == memory::HOST ? perm_map_hes_ : d_perm_map_hes_;
        break;
      case PERM_JAC_V:
        length      = nnz_jac_;
        apply_perm_ = memspace_ == memory::HOST ? perm_map_jac_ : d_perm_map_jac_;
        break;
      case PERM_JAC_TR_V:
        length      = nnz_jac_;
        apply_perm_ = memspace_ == memory::HOST ? perm_map_jac_tr_ : d_perm_map_jac_tr_;
        break;
      default:
        out::error() << "Valid arguments are PERM_V, REV_PERM_V, PERM_H_V, PERM_J_V, PERM_JT_V\n";
      }

      if (memspace_ == memory::HOST)
      {
        cpuImpl_->mapIdx(length, apply_perm_, old_val, new_val);
      }
      else
      {
        devImpl_->mapIdx(length, apply_perm_, old_val, new_val);
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
      if (perm_is_default_)
      {
        delete[] perm_;
      }
      delete[] rev_perm_;
      delete[] perm_map_hes_;
      delete[] perm_map_jac_;
      delete[] perm_map_jac_tr_;

      if (memspace_ == memory::DEVICE)
      {
        mem_.deleteOnDevice(d_perm_);
        mem_.deleteOnDevice(d_rev_perm_);
        mem_.deleteOnDevice(d_perm_map_hes_);
        mem_.deleteOnDevice(d_perm_map_jac_);
        mem_.deleteOnDevice(d_perm_map_jac_tr_);
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
      perm_            = new index_type[n_hes_];
      rev_perm_        = new index_type[n_hes_];
      perm_map_hes_    = new index_type[nnz_hes_];
      perm_map_jac_    = new index_type[nnz_jac_];
      perm_map_jac_tr_ = new index_type[nnz_jac_];

      if (memspace_ == memory::DEVICE)
      {
        mem_.allocateArrayOnDevice(&d_perm_, n_hes_);
        mem_.allocateArrayOnDevice(&d_rev_perm_, n_hes_);
        mem_.allocateArrayOnDevice(&d_perm_map_hes_, nnz_hes_);
        mem_.allocateArrayOnDevice(&d_perm_map_jac_, nnz_jac_);
        mem_.allocateArrayOnDevice(&d_perm_map_jac_tr_, nnz_jac_);
      }
    }
  } // namespace hykkt
} //  namespace ReSolve
