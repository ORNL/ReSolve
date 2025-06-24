/**
 * @file PermutationHandler.cpp
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief Implementation of PermutationHandler class.
 *
 *
 */
#include "PermutationHandler.hpp"

#include "cpuPermutationKernels.hpp"
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

    void PermutationHandler::mapIdx(int n, const int* perm, const double* old_val, double* new_val, memory::MemorySpace memspace)
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

    void PermutationHandler::insertionSort(int len, int* arr1, int* arr2, memory::MemorySpace memspace)
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

    void PermutationHandler::swap(int* arr1, int* arr2, int i, int j, memory::MemorySpace memspace)
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

    void PermutationHandler::quickSort(int* arr1, int* arr2, int low, int high, memory::MemorySpace memspace)
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

    void PermutationHandler::reversePerm(int n, const int* perm, int* rev_perm, memory::MemorySpace memspace)
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

    void PermutationHandler::makeVecMapC(int                 n,
                                         const int*          rows,
                                         const int*          cols,
                                         const int*          rev_perm,
                                         int*                perm_cols,
                                         int*                perm_map,
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

    void PermutationHandler::makeVecMapR(int                 n,
                                         const int*          rows,
                                         const int*          cols,
                                         const int*          perm,
                                         int*                perm_rows,
                                         int*                perm_cols,
                                         int*                perm_map,
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

    void PermutationHandler::makeVecMapRC(int                 n,
                                          const int*          rows,
                                          const int*          cols,
                                          const int*          perm,
                                          const int*          rev_perm,
                                          int*                perm_rows,
                                          int*                perm_cols,
                                          int*                perm_map,
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
