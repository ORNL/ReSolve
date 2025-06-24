#pragma once
/**
 * @file Permutation.hpp
 * @author Shaked Regev (regevs@ornl.gov)
 * @brief Declaration of hykkt::Permutation class
 *
 */
#include <resolve/hykkt/permutation/PermutationHandler.hpp>
#include <resolve/hykkt/permutation/PermutationKernelsImpl.hpp>

namespace ReSolve
{
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
      // constructors for each workspace type
      Permutation(int n_hes, int nnz_hes, int nnz_jac, memory::MemorySpace memspace);

      // destructor
      ~Permutation();

      void addHInfo(int* hes_i, int* hes_j);
      void addJInfo(int* jac_i, int* jac_j, int n_jac, int m_jac);
      void addJtInfo(int* jac_tr_i, int* jac_tr_j);
      void addPerm(int* custom_perm);
      void symAmd();
      void invertPerm();
      void vecMapRC(int* perm_i, int* perm_j);
      void vecMapC(int* perm_j);
      void vecMapR(int* perm_i, int* perm_j);
      void map_index(PermutationType permutation,
                     double*         old_val,
                     double*         new_val);
      void display_perm() const;

    private:
      void deleteWorkspace();
      void allocateWorkspace();

      //
      // member variables
      //

      PermutationHandler* permutationHandler_;
      memory::MemorySpace memspace_; ///< memory space for the permutation

      bool perm_is_default_ = true; ///< boolean if perm set custom

      int n_hes_;   ///< dimension of H
      int nnz_hes_; ///< nonzeros of H

      int n_jac_; ///< dimensions of J
      int m_jac_;
      int nnz_jac_; ///< nonzeros of J

      int* perm_;            ///< permutation of 2x2 system
      int* rev_perm_;        ///< reverse of permutation
      int* perm_map_hes_;    ///< mapping of permuted H
      int* perm_map_jac_;    ///< mapping of permuted J
      int* perm_map_jac_tr_; ///< mapping of permuted Jt

      int* hes_i_; ///< row offsets of csr storage of H
      int* hes_j_; ///< column pointers of csr storage of H

      int* jac_i_; ///< row offsets of csr storage of J
      int* jac_j_; ///< column pointers of csr storage of J

      int* jac_tr_i_; ///< row offsets of csr storage of J transpose
      int* jac_tr_j_; ///< column pointers of csr storage of J transpose

      ///< right hand side of 2x2 system
      double* rhs1_; ///< first block in vector
      double* rhs2_; ///< second block in vector
    };
  } // namespace hykkt
} // namespace ReSolve
