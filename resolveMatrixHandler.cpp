#include "resolveMatrixHandler.hpp"
#include <iostream>

namespace ReSolve
{
  //helper class
  indexPlusValue::indexPlusValue()
  {
    idx = 0;
    value = 0.0;
  }


  indexPlusValue::~indexPlusValue()
  {
  }

  void indexPlusValue::setIdx(resolveInt new_idx)
  {
    idx = new_idx;
  }

  void indexPlusValue::setValue(resolveReal new_value)
  {
    value = new_value;
  }

  resolveInt indexPlusValue::getIdx()
  {
    return idx;
  }

  resolveReal indexPlusValue::getValue()
  {
    return value;
  }

  //end of helper class


  resolveMatrixHandler::resolveMatrixHandler()
  {
  }

  resolveMatrixHandler::~resolveMatrixHandler()
  {
  }

  resolveMatrixHandler::resolveMatrixHandler(resolveLinAlgWorkspace* new_workspace)
  {
    workspace = new_workspace;
  }

  void resolveMatrixHandler::coo2csr(resolveMatrix* A, std::string memspace)
  {
    //this happens on the CPU not on the GPU
    //but will return whatever memspace requested.

    //count nnzs first

    resolveInt nnz_unpacked = 0;
    resolveInt nnz = A->getNnz();
    resolveInt n = A->getNumRows();
    bool symmetric = A->symmetric();
    bool expanded = A->expanded();

    resolveInt* nnz_counts =  new resolveInt[n];
    std::fill_n(nnz_counts, n, 0);
    resolveInt* coo_rows = A->getCooRowIndices("cpu");
    resolveInt* coo_cols = A->getCooColIndices("cpu");
    resolveReal* coo_vals = A->getCooValues("cpu");

    resolveInt* diag_control = new resolveInt[n]; //for DEDUPLICATION of the diagonal
    std::fill_n(diag_control, n, 0);
    resolveInt nnz_unpacked_no_duplicates = 0;
    resolveInt nnz_no_duplicates = nnz;


    //maybe check if they exist?
    for (resolveInt i = 0; i < nnz; ++i)
    {
      nnz_counts[coo_rows[i]]++;
      nnz_unpacked++;
      nnz_unpacked_no_duplicates++;
      if ((coo_rows[i] != coo_cols[i])&& (symmetric) && (!expanded))
      {
        nnz_counts[coo_cols[i]]++;
        nnz_unpacked++;
        nnz_unpacked_no_duplicates++;
      }
      if (coo_rows[i] == coo_cols[i]){
        if (diag_control[coo_rows[i]] > 0) {
          //duplicate
          nnz_unpacked_no_duplicates--;
          nnz_no_duplicates--;
        }
        diag_control[coo_rows[i]]++;
      }
    }
    A->setExpanded(true);
    A->setNnzExpanded(nnz_unpacked_no_duplicates);
    printf("original A nnz: %d true nnz %d expanded nnz %d de-duplicated unpacked nnz %d \n", nnz,nnz_no_duplicates, nnz_unpacked, nnz_unpacked_no_duplicates);
    resolveInt* csr_ia = new resolveInt[n+1];
    std::fill_n(csr_ia, n + 1, 0);
    resolveInt* csr_ja = new resolveInt[nnz_unpacked];
    resolveReal* csr_a = new resolveReal[nnz_unpacked];
    resolveInt* nnz_shifts = new resolveInt[n];
    std::fill_n(nnz_shifts, n , 0);

    indexPlusValue* tmp = new indexPlusValue[nnz_unpacked]; 

    csr_ia[0] = 0;

    for (resolveInt i = 1; i < n + 1; ++i){
      csr_ia[i] = csr_ia[i - 1] + nnz_counts[i - 1] - (diag_control[i-1] - 1);
    }

    int r, start;


    printf("\n\n");
    for (resolveInt i = 0; i < nnz; ++i){
      //which row
      r = coo_rows[i];
      start = csr_ia[r];

      if ((start + nnz_shifts[r]) > nnz_unpacked) {
        printf("index out of bounds 1: start %d nnz_shifts[%d] = %d \n", start, r, nnz_shifts[r]);
      }
      if ((r == coo_cols[i])&&(diag_control[r] > 1)) {//diagonal, and there are duplicates
        bool already_there = false;  
        for (resolveInt j = start; j < start + nnz_shifts[r]; ++j)
        {
          resolveInt c = tmp[j].getIdx();
          if (c == r) {
            resolveReal val = tmp[j].getValue();
            val += coo_vals[i];
            tmp[j].setValue(val);
            already_there = true;
            //printf("duplicate found, row %d, adding in place %d current value %f \n", c, j, val);
          }  
        }  
        if (!already_there){ // first time this duplicates appears

          tmp[start + nnz_shifts[r]].setIdx(coo_cols[i]);
          tmp[start + nnz_shifts[r]].setValue(coo_vals[i]);

          nnz_shifts[r]++;
        }
    } else {//not diagonal
      tmp[start + nnz_shifts[r]].setIdx(coo_cols[i]);
      tmp[start + nnz_shifts[r]].setValue(coo_vals[i]);
      nnz_shifts[r]++;

      if ((coo_rows[i] != coo_cols[i]) && (symmetric == 1))
      {
        r = coo_cols[i];
        start = csr_ia[r];

        if ((start + nnz_shifts[r]) > nnz_unpacked)
          printf("index out of bounds 2\n");
        tmp[start + nnz_shifts[r]].setIdx(coo_rows[i]);
        tmp[start + nnz_shifts[r]].setValue(coo_vals[i]);
        nnz_shifts[r]++;
      }
    }
  }
  //now sort whatever is inside rows

  for (int i = 0; i < n; ++i)
  {

    //now sorting (and adding 1)
    int colStart = csr_ia[i];
    int colEnd = csr_ia[i + 1];
    int length = colEnd - colStart;
    std::sort(&tmp[colStart],&tmp[colStart] + length);
  }

  for (resolveInt i = 0; i < nnz_unpacked; ++i)
  {
    csr_ja[i] = tmp[i].getIdx();
    csr_a[i] = tmp[i].getValue();
  }
#if 0
  for (int i = 0; i<n; ++i){
    printf("Row: %d \n", i);
    for (int j = csr_ia[i]; j<csr_ia[i+1]; ++j){
      printf("(%d %16.16f) ", csr_ja[j], csr_a[j]);
    }
    printf("\n");
  }
#endif
  A->setNnz(nnz_no_duplicates);
  if (memspace == "cpu"){
    A->updateCsr(csr_ia, csr_ja, csr_a, "cpu", "cpu");
  } else {
    if (memspace == "cuda"){      
      A->updateCsr(csr_ia, csr_ja, csr_a, "cpu", "cuda");
    } else {
      //display error
    }
  }
  delete [] nnz_counts;
  delete [] tmp;
  delete [] nnz_shifts;
  delete [] csr_ia;
  delete [] csr_ja;
  delete [] csr_a;
  delete [] diag_control; 
}

void resolveMatrixHandler::matvec(resolveMatrix* A, 
                                         resolveReal* x, 
                                         resolveReal* result, 
                                         resolveReal* alpha, 
                                         resolveReal* beta, 
                                         std::string memspace) 
{

  //result = alpha *A*x + beta * result
  if (memspace == "cuda" ){


    resolveLinAlgWorkspaceCUDA* workspaceCUDA = (resolveLinAlgWorkspaceCUDA*) workspace;

    cusparseDnVecDescr_t vecx = workspaceCUDA->getVecX();
    cusparseCreateDnVec(&vecx, A->getNumRows(), x , CUDA_R_64F);
    cusparseDnVecDescr_t vecAx = workspaceCUDA->getVecY();
    cusparseCreateDnVec(&vecAx, A->getNumRows(), result, CUDA_R_64F);
    cusparseSpMatDescr_t matA = workspaceCUDA->getSpmvMatrixDescriptor();

    void* buffer_spmv = workspaceCUDA->getSpmvBuffer();
    cusparseHandle_t handle_cusparse = workspaceCUDA->getCusparseHandle();
    if (!workspaceCUDA->matvecSetup()){
      //setup first, allocate, etc.
      size_t bufferSize = 0;
      resolveReal minusone = -1.0;
      resolveReal one = 1.0;
      cusparseCreateCsr(&matA, 
                        A->getNumRows(),
                        A->getNumColumns(),
                        A->getNnzExpanded(),
                        A->getCsrRowPointers("gpu"),
                        A->getCsrColIndices("gpu"),
                        A->getCsrValues("gpu"), 
                        CUSPARSE_INDEX_32I, 
                        CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO,
                        CUDA_R_64F);

      cusparseSpMV_bufferSize(handle_cusparse, 
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &minusone,
                              matA,
                              vecx,
                              &one,
                              vecAx,
                              CUDA_R_64F,
                              CUSPARSE_SPMV_CSR_ALG2, 
                              &bufferSize);

      cudaDeviceSynchronize();
      cudaMalloc(&buffer_spmv, bufferSize);


      workspaceCUDA->matvecSetupDone();
    } 

    cusparseSpMV(handle_cusparse,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,       
                 alpha, 
                 matA, 
                 vecx, 
                 beta, 
                 vecAx, 
                 CUDA_R_64F,
                 CUSPARSE_SPMV_CSR_ALG2, 
                 buffer_spmv);
  } else {
    std::cout<<"Not implemented (yet)"<<std::endl;
  }
}
}
