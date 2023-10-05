#include <algorithm>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/memoryUtils.hpp>
#include <resolve/LinAlgWorkspace.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include "MatrixHandler.hpp"

namespace ReSolve {
  // Create a shortcut name for Logger static class
  using out = io::Logger;

  //helper class
  indexPlusValue::indexPlusValue()
  {
    idx_ = 0;
    value_ = 0.0;
  }


  indexPlusValue::~indexPlusValue()
  {
  }

  void indexPlusValue::setIdx(index_type new_idx)
  {
    idx_ = new_idx;
  }

  void indexPlusValue::setValue(real_type new_value)
  {
    value_ = new_value;
  }

  index_type indexPlusValue::getIdx()
  {
    return idx_;
  }

  real_type indexPlusValue::getValue()
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

  int MatrixHandler::coo2csr(matrix::Coo* A_coo, matrix::Csr* A_csr, std::string memspace)
  {
    //this happens on the CPU not on the GPU
    //but will return whatever memspace requested.

    //count nnzs first

    index_type nnz_unpacked = 0;
    index_type nnz = A_coo->getNnz();
    index_type n = A_coo->getNumRows();
    bool symmetric = A_coo->symmetric();
    bool expanded = A_coo->expanded();

    index_type* nnz_counts =  new index_type[n];
    std::fill_n(nnz_counts, n, 0);
    index_type* coo_rows = A_coo->getRowData("cpu");
    index_type* coo_cols = A_coo->getColData("cpu");
    real_type* coo_vals = A_coo->getValues("cpu");

    index_type* diag_control = new index_type[n]; //for DEDUPLICATION of the diagonal
    std::fill_n(diag_control, n, 0);
    index_type nnz_unpacked_no_duplicates = 0;
    index_type nnz_no_duplicates = nnz;


    //maybe check if they exist?
    for (index_type i = 0; i < nnz; ++i)
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
    A_csr->setExpanded(true);
    A_csr->setNnzExpanded(nnz_unpacked_no_duplicates);
    index_type* csr_ia = new index_type[n+1];
    std::fill_n(csr_ia, n + 1, 0);
    index_type* csr_ja = new index_type[nnz_unpacked];
    real_type* csr_a = new real_type[nnz_unpacked];
    index_type* nnz_shifts = new index_type[n];
    std::fill_n(nnz_shifts, n , 0);

    indexPlusValue* tmp = new indexPlusValue[nnz_unpacked]; 

    csr_ia[0] = 0;

    for (index_type i = 1; i < n + 1; ++i){
      csr_ia[i] = csr_ia[i - 1] + nnz_counts[i - 1] - (diag_control[i-1] - 1);
    }

    int r, start;


    for (index_type i = 0; i < nnz; ++i){
      //which row
      r = coo_rows[i];
      start = csr_ia[r];

      if ((start + nnz_shifts[r]) > nnz_unpacked) {
        out::warning() << "index out of bounds (case 1) start: " << start << "nnz_shifts[" << r << "] = " << nnz_shifts[r] << std::endl;
      }
      if ((r == coo_cols[i]) && (diag_control[r] > 1)) {//diagonal, and there are duplicates
        bool already_there = false;  
        for (index_type j = start; j < start + nnz_shifts[r]; ++j)
        {
          index_type c = tmp[j].getIdx();
          if (c == r) {
            real_type val = tmp[j].getValue();
            val += coo_vals[i];
            tmp[j].setValue(val);
            already_there = true;
            out::warning() << " duplicate found, row " << c << " adding in place " << j << " current value: " << val << std::endl;
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
            out::warning() << "index out of bounds (case 2) start: " << start << "nnz_shifts[" << r << "] = " << nnz_shifts[r] << std::endl;
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

    for (index_type i = 0; i < nnz_unpacked; ++i)
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
    A_csr->setNnz(nnz_no_duplicates);
    if (memspace == "cpu"){
      A_csr->updateData(csr_ia, csr_ja, csr_a, "cpu", "cpu");
    } else {
      if (memspace == "cuda"){      
        A_csr->updateData(csr_ia, csr_ja, csr_a, "cpu", "cuda");
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

    return 0;
  }

  int MatrixHandler::matvec(matrix::Sparse* Ageneric, 
                            vector_type* vec_x, 
                            vector_type* vec_result, 
                            const real_type* alpha, 
                            const real_type* beta,
                            std::string matrixFormat, 
                            std::string memspace) 
  {
    using namespace constants;
    int error_sum = 0;
    if (matrixFormat == "csr"){
      matrix::Csr* A = (matrix::Csr*) Ageneric;
      cusparseStatus_t status;
      //result = alpha *A*x + beta * result
      if (memspace == "cuda" ){
        // std::cout << "Matvec on NVIDIA GPU ...\n";
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
                                     A->getRowData("cuda"),
                                     A->getColData("cuda"),
                                     A->getValues("cuda"), 
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

          status = cusparseSpMV_bufferSize(handle_cusparse, 
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &MINUSONE,
                                           matA,
                                           vecx,
                                           &ONE,
                                           vecAx,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2, 
                                           &bufferSize);
          error_sum += status;
          deviceSynchronize();
          allocateBufferOnDevice(&buffer_spmv, bufferSize);
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
        deviceSynchronize();
        if (status)
          out::error() << "Matvec status: " << status 
                       << "Last error code: " << getLastDeviceError() << std::endl;
        vec_result->setDataUpdated("cuda");

        cusparseDestroyDnVec(vecx);
        cusparseDestroyDnVec(vecAx);
        return error_sum;
      } else {
        if (memspace == "cpu"){
          //cpu csr matvec
          index_type* ia = A->getRowData("cpu");
          index_type* ja = A->getColData("cpu");
          real_type* a = A->getValues("cpu");

          real_type* x_data = vec_x->getData("cpu");
          real_type* result_data = vec_result->getData("cpu");
          real_type sum;
          real_type y, t, c;
          //Kahan algorithm for stability; Kahan-Babushka version didnt make a difference   
          for (int i = 0; i < A->getNumRows(); ++i) {
            sum = 0.0;
            c = 0.0;
            for (int j = ia[i]; j < ia[i+1]; ++j) { 
              y =  ( a[j] * x_data[ja[j]]) - c;
              t = sum + y;
              c = (t - sum) - y;
              sum = t;
              //  sum += ( a[j] * x_data[ja[j]]);
            }
            sum *= (*alpha);
            result_data[i] = result_data[i]*(*beta) + sum;
          } 
          vec_result->setDataUpdated("cpu");
          return 0;
        } else {
          out::error() << "Not implemented (yet)" << std::endl;

          return 1;
        }
      }
    } else {

      out::error() << "Not implemented (yet), only csr matvec is implemented" << std::endl;
      return 1;
    }
  }


  int MatrixHandler::csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr, std::string memspace)
  {
    //it ONLY WORKS WITH CUDA
    index_type error_sum = 0;
    if (memspace == "cuda") { 
      LinAlgWorkspaceCUDA* workspaceCUDA = (LinAlgWorkspaceCUDA*) workspace_;

      A_csr->allocateMatrixData("cuda");
      index_type n = A_csc->getNumRows();
      index_type m = A_csc->getNumRows();
      index_type nnz = A_csc->getNnz();
      size_t bufferSize;
      void* d_work;
      cusparseStatus_t status = cusparseCsr2cscEx2_bufferSize(workspaceCUDA->getCusparseHandle(),
                                                              n, 
                                                              m, 
                                                              nnz, 
                                                              A_csc->getValues("cuda"), 
                                                              A_csc->getColData("cuda"), 
                                                              A_csc->getRowData("cuda"), 
                                                              A_csr->getValues("cuda"), 
                                                              A_csr->getRowData("cuda"),
                                                              A_csr->getColData("cuda"), 
                                                              CUDA_R_64F, 
                                                              CUSPARSE_ACTION_NUMERIC,
                                                              CUSPARSE_INDEX_BASE_ZERO, 
                                                              CUSPARSE_CSR2CSC_ALG1, 
                                                              &bufferSize);
      error_sum += status;
      allocateBufferOnDevice(&d_work, bufferSize);
      status = cusparseCsr2cscEx2(workspaceCUDA->getCusparseHandle(),
                                  n, 
                                  m, 
                                  nnz, 
                                  A_csc->getValues("cuda"), 
                                  A_csc->getColData("cuda"), 
                                  A_csc->getRowData("cuda"), 
                                  A_csr->getValues("cuda"), 
                                  A_csr->getRowData("cuda"),
                                  A_csr->getColData("cuda"), 
                                  CUDA_R_64F,
                                  CUSPARSE_ACTION_NUMERIC,
                                  CUSPARSE_INDEX_BASE_ZERO,
                                  CUSPARSE_CSR2CSC_ALG1,
                                  d_work);
      error_sum += status;
      return error_sum;
      deleteOnDevice(d_work);
    } else { 
      out::error() << "Not implemented (yet)" << std::endl;
      return -1;
    } 


  }

} // namespace ReSolve
