/**
 * @file PermutationHandler.cpp
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief Implementation of PermutationHandler class.
 *
 *
 */
#include "PermutationHandler.hpp"

#include "CpuPermutationKernels.hpp"
#ifdef RESOLVE_USE_CUDA
#include "CudaPermutationKernels.hpp"
#endif
#ifdef RESOLVE_USE_HIP
#include "HipPermutationKernels.hpp"
#endif

namespace ReSolve
{
  namespace hykkt
  {
    PermutationHandler::PermutationHandler()
    {
      cpuImpl_ = new CpuPermutationKernels();
#ifdef RESOLVE_USE_CUDA
      devImpl_       = new CudaPermutationKernels();
      isCudaEnabled_ = true;
      isHipEnabled_  = false;
#endif
#ifdef RESOLVE_USE_HIP
      devImpl_       = new HipPermutationKernels();
      isHipEnabled_  = true;
      isCudaEnabled_ = false;
#endif
      if (!isCudaEnabled_ && !isHipEnabled_)
      {
        devImpl_ = nullptr;
      }
    }

    PermutationHandler::~PermutationHandler()
    {
      delete cpuImpl_;
      if (devImpl_ != nullptr)
      {
        delete devImpl_;
      }
    }

    void PermutationHandler::mapIdx(index_type n, const index_type* perm, const real_type* old_val, real_type* new_val, memory::MemorySpace memspace)
    {
      if (memspace == memory::HOST)
      {
        cpuImpl_->mapIdx(n, perm, old_val, new_val);
      }
      else
      {
        devImpl_->mapIdx(n, perm, old_val, new_val);
      }
    }

    void PermutationHandler::insertionSort(index_type len, index_type* arr1, index_type* arr2, memory::MemorySpace memspace)
    {
      if (memspace == memory::HOST)
      {
        cpuImpl_->insertionSort(len, arr1, arr2);
      }
      else
      {
        devImpl_->insertionSort(len, arr1, arr2);
      }
    }

    void PermutationHandler::swap(index_type* arr1, index_type* arr2, index_type i, index_type j, memory::MemorySpace memspace)
    {
      if (memspace == memory::HOST)
      {
        cpuImpl_->swap(arr1, arr2, i, j);
      }
      else
      {
        devImpl_->swap(arr1, arr2, i, j);
      }
    }

    void PermutationHandler::quickSort(index_type* arr1, index_type* arr2, index_type low, index_type high, memory::MemorySpace memspace)
    {
      if (memspace == memory::HOST)
      {
        cpuImpl_->quickSort(arr1, arr2, low, high);
      }
      else
      {
        devImpl_->quickSort(arr1, arr2, low, high);
      }
    }

    void PermutationHandler::reversePerm(index_type n, const index_type* perm, index_type* rev_perm, memory::MemorySpace memspace)
    {
      if (memspace == memory::HOST)
      {
        cpuImpl_->reversePerm(n, perm, rev_perm);
      }
      else
      {
        devImpl_->reversePerm(n, perm, rev_perm);
      }
    }

    void PermutationHandler::makeVecMapC(index_type          n,
                                         const index_type*   rows,
                                         const index_type*   cols,
                                         const index_type*   rev_perm,
                                         index_type*         perm_cols,
                                         index_type*         perm_map,
                                         memory::MemorySpace memspace)
    {
      if (memspace == memory::HOST)
      {
        cpuImpl_->makeVecMapC(n, rows, cols, rev_perm, perm_cols, perm_map);
      }
      else
      {
        devImpl_->makeVecMapC(n, rows, cols, rev_perm, perm_cols, perm_map);
      }
    }

    void PermutationHandler::makeVecMapR(index_type          n,
                                         const index_type*   rows,
                                         const index_type*   cols,
                                         const index_type*   perm,
                                         index_type*         perm_rows,
                                         index_type*         perm_cols,
                                         index_type*         perm_map,
                                         memory::MemorySpace memspace)
    {
      if (memspace == memory::HOST)
      {
        cpuImpl_->makeVecMapR(n, rows, cols, perm, perm_rows, perm_cols, perm_map);
      }
      else
      {
        devImpl_->makeVecMapR(n, rows, cols, perm, perm_rows, perm_cols, perm_map);
      }
    }

    void PermutationHandler::makeVecMapRC(index_type          n,
                                          const index_type*   rows,
                                          const index_type*   cols,
                                          const index_type*   perm,
                                          const index_type*   rev_perm,
                                          index_type*         perm_rows,
                                          index_type*         perm_cols,
                                          index_type*         perm_map,
                                          memory::MemorySpace memspace)
    {
      if (memspace == memory::HOST)
      {
        cpuImpl_->makeVecMapRC(n, rows, cols, perm, rev_perm, perm_rows, perm_cols, perm_map);
      }
      else
      {
        devImpl_->makeVecMapRC(n, rows, cols, perm, rev_perm, perm_rows, perm_cols, perm_map);
      }
    }
  } // namespace hykkt
} // namespace ReSolve
