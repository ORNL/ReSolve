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

    class PermutationHandler
    {
    public:
      PermutationHandler(LinAlgWorkspaceCpu* workspaceCpu);
#ifdef RESOLVE_USE_CUDA
      PermutationHandler(LinAlgWorkspaceCUDA* workspaceCuda);
#endif
#ifdef RESOLVE_USE_HIP
      PermutationHandler(LinAlgWorkspaceHIP* workspaceHip);
#endif
      ~PermutationHandler();

      void        mapIdx(int                 n,
                         const int*          perm,
                         const double*       old_val,
                         double*             new_val,
                         memory::MemorySpace memspace);
      void        selectionSort(int len, int* arr1, int* arr2, memory::MemorySpace memspace);
      inline void swap(int* arr1, int* arr2, int i, int j, memory::MemorySpace memspace);
      void        quickSort(int* arr1, int* arr2, int low, int high, memory::MemorySpace memspace);
      void        insertionSort(int n, int* arr1, int* arr2, memory::MemorySpace memspace);
      void        reversePerm(int n, const int* perm, int* rev_perm, memory::MemorySpace memspace);
      void        makeVecMapC(int                 n,
                              const int*          rows,
                              const int*          cols,
                              const int*          rev_perm,
                              int*                perm_cols,
                              int*                perm_map,
                              memory::MemorySpace memspace);
      void        makeVecMapR(int                 n,
                              const int*          rows,
                              const int*          cols,
                              const int*          perm,
                              int*                perm_rows,
                              int*                perm_cols,
                              int*                perm_map,
                              memory::MemorySpace memspace);
      void        makeVecMapRC(int                 n,
                               const int*          rows,
                               const int*          cols,
                               const int*          perm,
                               const int*          rev_perm,
                               int*                perm_rows,
                               int*                perm_cols,
                               int*                perm_map,
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
