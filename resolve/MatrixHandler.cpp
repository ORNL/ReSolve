#include "MatrixHandler.hpp"
#include <iostream>

namespace ReSolve
{
  //helper class
  indexPlusValue::indexPlusValue()
  {
    idx_ = 0;
    value_ = 0.0;
  }


  indexPlusValue::~indexPlusValue()
  {
  }

  void indexPlusValue::setIdx(Int new_idx)
  {
    idx_ = new_idx;
  }

  void indexPlusValue::setValue(Real new_value)
  {
    value_ = new_value;
  }

  Int indexPlusValue::getIdx()
  {
    return idx_;
  }

  Real indexPlusValue::getValue()
  {
    return value_;
  }
  //end of helper class

  MatrixHandler::MatrixHandler()
  {
    this->new_matrix_ = true;
    this->values_changed_ = true;
  }

  MatrixHandler::~MatrixHandler()
  {
  }

  MatrixHandler::MatrixHandler(LinAlgWorkspace* new_workspace)
  {
    workspace_ = new_workspace;
  }

  bool MatrixHandler::getValuesChanged()
  {
    return this->values_changed_;
  }

  void MatrixHandler::setValuesChanged(bool toWhat)
  {
    this->values_changed_ = toWhat;
  }

  void MatrixHandler::coo2csr(Matrix* A, std::string memspace)
  {
    //this happens on the CPU not on the GPU
    //but will return whatever memspace requested.

    //count nnzs first

    Int nnz_unpacked = 0;
    Int nnz = A->getNnz();
    Int n = A->getNumRows();
    bool symmetric = A->symmetric();
    bool expanded = A->expanded();

    Int* nnz_counts =  new Int[n];
    std::fill_n(nnz_counts, n, 0);
    Int* coo_rows = A->getCooRowIndices("cpu");
    Int* coo_cols = A->getCooColIndices("cpu");
    Real* coo_vals = A->getCooValues("cpu");

    Int* diag_control = new Int[n]; //for DEDUPLICATION of the diagonal
    std::fill_n(diag_control, n, 0);
    Int nnz_unpacked_no_duplicates = 0;
    Int nnz_no_duplicates = nnz;


    //maybe check if they exist?
    for (Int i = 0; i < nnz; ++i)
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
    Int* csr_ia = new Int[n+1];
    std::fill_n(csr_ia, n + 1, 0);
    Int* csr_ja = new Int[nnz_unpacked];
    Real* csr_a = new Real[nnz_unpacked];
    Int* nnz_shifts = new Int[n];
    std::fill_n(nnz_shifts, n , 0);

    indexPlusValue* tmp = new indexPlusValue[nnz_unpacked]; 

    csr_ia[0] = 0;

    for (Int i = 1; i < n + 1; ++i){
      csr_ia[i] = csr_ia[i - 1] + nnz_counts[i - 1] - (diag_control[i-1] - 1);
    }

    int r, start;


    for (Int i = 0; i < nnz; ++i){
      //which row
      r = coo_rows[i];
      start = csr_ia[r];

      if ((start + nnz_shifts[r]) > nnz_unpacked) {
        printf("index out of bounds 1: start %d nnz_shifts[%d] = %d \n", start, r, nnz_shifts[r]);
      }
      if ((r == coo_cols[i]) && (diag_control[r] > 1)) {//diagonal, and there are duplicates
        bool already_there = false;  
        for (Int j = start; j < start + nnz_shifts[r]; ++j)
        {
          Int c = tmp[j].getIdx();
          if (c == r) {
            Real val = tmp[j].getValue();
            val += coo_vals[i];
            tmp[j].setValue(val);
            already_there = true;
            //printf("duplicate found, row %d, adding in place %d current value_ %f \n", c, j, val);
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

    for (Int i = 0; i < nnz_unpacked; ++i)
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

  int MatrixHandler::matvec(Matrix* A, 
                            Vector* vec_x, 
                            Vector* vec_result, 
                            Real* alpha, 
                            Real* beta, 
                            std::string memspace) 
  {
    int error_sum = 0;
    cusparseStatus_t status;
    //result = alpha *A*x + beta * result
    if (memspace == "cuda" ){

      LinAlgWorkspaceCUDA* workspaceCUDA = (LinAlgWorkspaceCUDA*) workspace_;
      cusparseDnVecDescr_t vecx = workspaceCUDA->getVecX();
      //printf("is vec_x NULL? %d\n", vec_x->getData("cuda") == nullptr);
      //printf("is vec_result NULL? %d\n", vec_result->getData("cuda") == nullptr);
      cusparseCreateDnVec(&vecx, A->getNumRows(), vec_x->getData("cuda"), CUDA_R_64F);


      cusparseDnVecDescr_t vecAx = workspaceCUDA->getVecY();
      cusparseCreateDnVec(&vecAx, A->getNumRows(), vec_result->getData("cuda"), CUDA_R_64F);

      cusparseSpMatDescr_t matA = workspaceCUDA->getSpmvMatrixDescriptor();

      void* buffer_spmv = workspaceCUDA->getSpmvBuffer();
      cusparseHandle_t handle_cusparse = workspaceCUDA->getCusparseHandle();
      if (values_changed_){ 
        status = cusparseCreateCsr(&matA, 
                                   A->getNumRows(),
                                   A->getNumColumns(),
                                   A->getNnzExpanded(),
                                   A->getCsrRowPointers("cuda"),
                                   A->getCsrColIndices("cuda"),
                                   A->getCsrValues("cuda"), 
                                   CUSPARSE_INDEX_32I, 
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_64F);
        error_sum += status;
        values_changed_ = false;
      }
      if (!workspaceCUDA->matvecSetup()){
        //setup first, allocate, etc.
        size_t bufferSize = 0;
        Real minusone = -1.0;
        Real one = 1.0;

        status = cusparseSpMV_bufferSize(handle_cusparse, 
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &minusone,
                                         matA,
                                         vecx,
                                         &one,
                                         vecAx,
                                         CUDA_R_64F,
                                         CUSPARSE_SPMV_CSR_ALG2, 
                                         &bufferSize);
        error_sum += status;
        cudaDeviceSynchronize();
        cudaMalloc(&buffer_spmv, bufferSize);
        workspaceCUDA->setSpmvMatrixDescriptor(matA);
        workspaceCUDA->setSpmvBuffer(buffer_spmv);

        workspaceCUDA->matvecSetupDone();
      } 

      status = cusparseSpMV(handle_cusparse,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,       
                            alpha, 
                            matA, 
                            vecx, 
                            beta, 
                            vecAx, 
                            CUDA_R_64F,
                            CUSPARSE_SPMV_CSR_ALG2, 
                            buffer_spmv);
      error_sum += status;
      cudaDeviceSynchronize();
      if (status) printf("Matvec status: %d Last ERROR %d \n", status,  	cudaGetLastError() );
      vec_result->setDataUpdated("cuda");

      cusparseDestroyDnVec(vecx);
      cusparseDestroyDnVec(vecAx);
      return error_sum;
    } else {
      std::cout<<"Not implemented (yet)"<<std::endl;
      return 1;
    }
  }

  void MatrixHandler::csc2csr(Matrix* A, std::string memspace)
  {
    //it ONLY WORKS WITH CUDA
    if (memspace == "cuda") { 
      LinAlgWorkspaceCUDA* workspaceCUDA = (LinAlgWorkspaceCUDA*) workspace_;

      A->allocateCsr("cuda");
      Int n = A->getNumRows();
      Int m = A->getNumRows();
      Int nnz = A->getNnz();

      size_t bufferSize;
      void* d_work;
      cusparseStatus_t status = cusparseCsr2cscEx2_bufferSize(workspaceCUDA->getCusparseHandle(),
                                                              n, 
                                                              m, 
                                                              nnz, 
                                                              A->getCscValues("cuda"), 
                                                              A->getCscColPointers("cuda"), 
                                                              A->getCscRowIndices("cuda"), 
                                                              A->getCsrValues("cuda"), 
                                                              A->getCsrRowPointers("cuda"),
                                                              A->getCsrColIndices("cuda"), 
                                                              CUDA_R_64F, 
                                                              CUSPARSE_ACTION_NUMERIC,
                                                              CUSPARSE_INDEX_BASE_ZERO, 
                                                              CUSPARSE_CSR2CSC_ALG1, 
                                                              &bufferSize);

      cudaMalloc((void**)&d_work, bufferSize);
      status = cusparseCsr2cscEx2(workspaceCUDA->getCusparseHandle(),
                                  n, 
                                  m, 
                                  nnz, 
                                  A->getCscValues("cuda"), 
                                  A->getCscColPointers("cuda"), 
                                  A->getCscRowIndices("cuda"), 
                                  A->getCsrValues("cuda"), 
                                  A->getCsrRowPointers("cuda"),
                                  A->getCsrColIndices("cuda"),                             
                                  CUDA_R_64F,
                                  CUSPARSE_ACTION_NUMERIC,
                                  CUSPARSE_INDEX_BASE_ZERO,
                                  CUSPARSE_CSR2CSC_ALG1,
                                  d_work);

      cudaFree(d_work);
    } else { 
      std::cout<<"Not implemented (yet)"<<std::endl;
    } 


  }
}
