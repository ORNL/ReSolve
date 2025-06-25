/**
 * @file PermutationHandler.hpp
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief Declaration of PermutationHandler class.
 *
 *
 */
#pragma once
#include "PermutationKernelsImpl.hpp"
#include <resolve/MemoryUtils.hpp>
#include <resolve/workspace/LinAlgWorkspaceCpu.hpp>

namespace ReSolve
{
  class LinAlgWorkspaceCpu;
  class LinAlgWorkspaceCUDA;
  class LinAlgWorkspaceHIP;
} // namespace ReSolve

namespace ReSolve
{
  namespace hykkt
  {

    /**
     * @class PermutationHandler
     *
     * @brief This class is a wrapper for the kernels used for the hykkt::Permutation
     *        operations. 
     *
     */
    class PermutationHandler
    {
    public:
      PermutationHandler();
      ~PermutationHandler();

      void        mapIdx(index_type          n,
                         const index_type*   perm,
                         const real_type*    old_val,
                         real_type*          new_val,
                         memory::MemorySpace memspace);
      void        selectionSort(index_type len, index_type* arr1, index_type* arr2, memory::MemorySpace memspace);
      inline void swap(index_type* arr1, index_type* arr2, index_type i, index_type j, memory::MemorySpace memspace);
      void        quickSort(index_type* arr1, index_type* arr2, index_type low, index_type high, memory::MemorySpace memspace);
      void        insertionSort(index_type n, index_type* arr1, index_type* arr2, memory::MemorySpace memspace);
      void        reversePerm(index_type n, const index_type* perm, index_type* rev_perm, memory::MemorySpace memspace);
      void        makeVecMapC(index_type          n,
                              const index_type*   rows,
                              const index_type*   cols,
                              const index_type*   rev_perm,
                              index_type*         perm_cols,
                              index_type*         perm_map,
                              memory::MemorySpace memspace);
      void        makeVecMapR(index_type          n,
                              const index_type*   rows,
                              const index_type*   cols,
                              const index_type*   perm,
                              index_type*         perm_rows,
                              index_type*         perm_cols,
                              index_type*         perm_map,
                              memory::MemorySpace memspace);
      void        makeVecMapRC(index_type          n,
                               const index_type*   rows,
                               const index_type*   cols,
                               const index_type*   perm,
                               const index_type*   rev_perm,
                               index_type*         perm_rows,
                               index_type*         perm_cols,
                               index_type*         perm_map,
                               memory::MemorySpace memspace);

      bool getIsCudaEnabled()
      {
        return isCudaEnabled_;
      }

      bool getIsHipEnabled()
      {
        return isHipEnabled_;
      }

    private:
      PermutationKernelsImpl* cpuImpl_;
      PermutationKernelsImpl* devImpl_;

      bool isCudaEnabled_; ///< true if CUDA implementation is instantiated
      bool isHipEnabled_;  ///< true if HIP implementation is instantiated
    };
  } // namespace hykkt
} // namespace ReSolve
