#pragma once
/**
 * @file Permutation.hpp
 * @author Shaked Regev (regevs@ornl.gov)
 * @brief Declaration of hykkt::Permutation class
 *
 */

#include <resolve/Common.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>

#include "PermutationKernelsImpl.hpp"
#include "CpuPermutationKernels.hpp"
#ifdef RESOLVE_USE_CUDA
#include "CudaPermutationKernels.hpp"
#endif
#ifdef RESOLVE_USE_HIP
#include "HipPermutationKernels.hpp"
#endif

namespace ReSolve
{
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;

  namespace hykkt
  {
    enum PermutationType
    {
      PERM_V,
      REV_PERM_V,
      PERM_HES_V,
      PERM_JAC_V,
      PERM_JAC_TR_V
    };

    /**
     * @class Permutation
     *
     * @brief Creates a permutation of the 2x2 system.
     *
     * This class creates a permutation of the 2x2 system, which is obtained
     * by block-Gauss elimination in the 4x4 system. The permutation is
     * based on the Symmetric Approximate Minimum Degree algorithm to minimize
     * fill in for H and maps the permutation to the matrices H, J, and J
     * transpose.
     *
     */
    class Permutation
    {
    public:
      Permutation(index_type n_hes, index_type n_jac, index_type nnz_hes, index_type nnz_jac, memory::MemorySpace memspace);
      ~Permutation();

      void addMatrixInfo(matrix::Csr* hes, matrix::Csr* jac, matrix::Csr* jac_tr);
      void addCustomPerm(index_type* custom_perm);
      void symAmd();
      void invertPerm();
      void vecMapRC(index_type* perm_i, index_type* perm_j);
      void vecMapC(index_type* perm_j);
      void vecMapR(index_type* perm_i, index_type* perm_j);
      void mapIndex(PermutationType permutation,
                    real_type*      old_val,
                    real_type*      new_val);

    private:
      void deleteWorkspace();
      void allocateWorkspace();

      MemoryHandler       mem_;      ///< memory handler for the permutation
      memory::MemorySpace memspace_; ///< memory space for the permutation

      PermutationKernelsImpl* cpuImpl_; ///< pointer to the implementation of the permutation kernels
      PermutationKernelsImpl* devImpl_;  ///< pointer to the device implementation of the permutation kernels

      index_type n_hes_;   ///< dimension of H
      index_type nnz_hes_; ///< nonzeros of H

      index_type n_jac_; ///< dimensions of J
      index_type m_jac_;
      index_type nnz_jac_; ///< nonzeros of J

      bool        perm_is_default_;
      index_type* perm_;            ///< permutation of 2x2 system
      index_type* rev_perm_;        ///< reverse of permutation
      index_type* perm_map_hes_;    ///< mapping of permuted H
      index_type* perm_map_jac_;    ///< mapping of permuted J
      index_type* perm_map_jac_tr_; ///< mapping of permuted Jt

      index_type* d_perm_;
      index_type* d_rev_perm_;
      index_type* d_perm_map_hes_;
      index_type* d_perm_map_jac_;
      index_type* d_perm_map_jac_tr_;

      index_type* hes_i_; ///< row offsets of csr storage of H
      index_type* hes_j_; ///< column pointers of csr storage of H

      index_type* jac_i_; ///< row offsets of csr storage of J
      index_type* jac_j_; ///< column pointers of csr storage of J

      index_type* jac_tr_i_; ///< row offsets of csr storage of J transpose
      index_type* jac_tr_j_; ///< column pointers of csr storage of J transpose

      ///< right hand side of 2x2 system
      real_type* rhs1_; ///< first block in vector
      real_type* rhs2_; ///< second block in vector
    };
  } // namespace hykkt
} // namespace ReSolve
