
/**
 * @file cusolver_defs.hpp
 *
 * @author Kasia Swirydowicz <kasia.Swirydowicz@pnnl.gov>, PNNL
 * 
 * Contains prototypes of cuSOLVER functions not in public API.
 *
 */

#ifndef CUSOLVERDEFS_H
#define CUSOLVERDEFS_H

#include "cusparse.h"
#include "cusolverSp.h"
#include <assert.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cusolverSp_LOWLEVEL_PREVIEW.h"

#include "cusolverRf.h"

extern "C" {
  /*
   * prototype not in public header file 
   */
  struct csrgluInfo;
  typedef struct csrgluInfo *csrgluInfo_t;

  cusolverStatus_t CUSOLVERAPI
  cusolverSpCreateGluInfo(csrgluInfo_t *info);

  cusolverStatus_t CUSOLVERAPI
  cusolverSpDestroyGluInfo(csrgluInfo_t info);

  cusolverStatus_t CUSOLVERAPI
  cusolverSpDgluSetup(cusolverSpHandle_t handle,
                      int m,
                      /* A can be base-0 or base-1 */
                      int nnzA,
                      const cusparseMatDescr_t descrA,
                      const int* h_csrRowPtrA,
                      const int* h_csrColIndA,
                      const int* h_P, /* base-0 */
                      const int* h_Q, /* base-0 */
                      /* M can be base-0 or base-1 */
                      int nnzM,
                      const cusparseMatDescr_t descrM,
                      const int* h_csrRowPtrM,
                      const int* h_csrColIndM,
                      csrgluInfo_t info);

  cusolverStatus_t CUSOLVERAPI
  cusolverSpDgluBufferSize(cusolverSpHandle_t handle,
                           csrgluInfo_t info,
                           size_t* pBufferSize);

  cusolverStatus_t CUSOLVERAPI
  cusolverSpDgluAnalysis(cusolverSpHandle_t handle,
                         csrgluInfo_t info,
                         void* workspace);

  cusolverStatus_t CUSOLVERAPI
  cusolverSpDgluReset(cusolverSpHandle_t handle,
                      int m,
                      /* A is original matrix */
                      int nnzA,
                      const cusparseMatDescr_t descr_A,
                      const double* d_csrValA,
                      const int* d_csrRowPtrA,
                      const int* d_csrColIndA,
                      csrgluInfo_t info);

  cusolverStatus_t CUSOLVERAPI
  cusolverSpDgluFactor(cusolverSpHandle_t handle,
                       csrgluInfo_t info,
                       void *workspace);

  cusolverStatus_t CUSOLVERAPI
  cusolverSpDgluSolve(cusolverSpHandle_t handle,
                      int m,
                      /* A is original matrix */
                      int nnzA,
                      const cusparseMatDescr_t descr_A,
                      const double *d_csrValA,
                      const int* d_csrRowPtrA,
                      const int* d_csrColIndA,
                      const double* d_b0, /* right hand side */
                      double* d_x, /* left hand side */
                      int* ite_refine_succ,
                      double* r_nrm_inf_ptr,
                      csrgluInfo_t info,
                      void* workspace);

  cusolverStatus_t CUSOLVERAPI 
    cusolverSpDnrm_inf(cusolverSpHandle_t handle,
                      int n,
                      const double *x,
                      double* result, /* |x|_inf, host */
                      void* d_work  /* at least 8192 bytes */
                     );


} // extern "C"
#endif // CUSOLVERDEFS_H
