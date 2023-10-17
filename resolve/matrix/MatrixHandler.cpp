#include <algorithm>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/workspace/LinAlgWorkspaceFactory.hpp>
#include "MatrixHandler.hpp"
#include "MatrixHandlerCpu.hpp"
#include "MatrixHandlerCuda.hpp"

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
    cpuImpl_  = new MatrixHandlerCpu();
    cudaImpl_ = new MatrixHandlerCuda();
  }

  MatrixHandler::~MatrixHandler()
  {
  }

  MatrixHandler::MatrixHandler(LinAlgWorkspace* new_workspace)
  {
    workspace_ = new_workspace;
    cpuImpl_  = new MatrixHandlerCpu(new_workspace);
    cudaImpl_ = new MatrixHandlerCuda(new_workspace);
  }

  void MatrixHandler::setValuesChanged(bool toWhat)
  {
    this->values_changed_ = toWhat;
    cpuImpl_->setValuesChanged(values_changed_);
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

  /**
   * @brief Matrix vector product method  result = alpha *A*x + beta * result
   * @param A 
   * @param vec_x 
   * @param vec_result 
   * @param[in] alpha 
   * @param[in] beta 
   * @param[in] matrixFormat 
   * @param[in] memspace 
   * @return result := alpha * A * x + beta * result
   */
  int MatrixHandler::matvec(matrix::Sparse* A, 
                            vector_type* vec_x, 
                            vector_type* vec_result, 
                            const real_type* alpha, 
                            const real_type* beta,
                            std::string matrixFormat, 
                            std::string memspace)
  {
    if (memspace == "cuda" ) {
      return cudaImpl_->matvec(A, vec_x, vec_result, alpha, beta, matrixFormat);
    } else if (memspace == "cpu") {
        return cpuImpl_->matvec(A, vec_x, vec_result, alpha, beta, matrixFormat);
    } else {
        out::error() << "Support for device " << memspace << " not implemented (yet)" << std::endl;
        return 1;
    }
  }


  int MatrixHandler::csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr, std::string memspace)
  {
    index_type error_sum = 0;
    if (memspace == "cuda") { 
      return cudaImpl_->csc2csr(A_csc, A_csr);
    } else if (memspace == "cpu") { 
      out::warning() << "Using untested csc2csr on CPU ..." << std::endl;
      return cpuImpl_->csc2csr(A_csc, A_csr);
    } else {
      out::error() << "csc2csr not implemented for " << memspace << " device." << std::endl;
      return -1;
    }
  }

} // namespace ReSolve
