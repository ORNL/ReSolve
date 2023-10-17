#include <algorithm>
#include <cassert>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include "MatrixHandlerCpu.hpp"

namespace ReSolve {
  // Create a shortcut name for Logger static class
  using out = io::Logger;

  MatrixHandlerCpu::MatrixHandlerCpu()
  {
    // new_matrix_ = true;
    // values_changed_ = true;
  }

  MatrixHandlerCpu::~MatrixHandlerCpu()
  {
  }

  MatrixHandlerCpu::MatrixHandlerCpu(LinAlgWorkspace* new_workspace)
  {
    workspace_ = new_workspace;
  }

  // bool MatrixHandlerCpu::getValuesChanged()
  // {
  //   return this->values_changed_;
  // }

  void MatrixHandlerCpu::setValuesChanged(bool values_changed)
  {
    values_changed_ = values_changed;
  }

//   int MatrixHandlerCpu::coo2csr(matrix::Coo* A_coo, matrix::Csr* A_csr, std::string memspace)
//   {
//     //this happens on the CPU not on the GPU
//     //but will return whatever memspace requested.

//     //count nnzs first

//     index_type nnz_unpacked = 0;
//     index_type nnz = A_coo->getNnz();
//     index_type n = A_coo->getNumRows();
//     bool symmetric = A_coo->symmetric();
//     bool expanded = A_coo->expanded();

//     index_type* nnz_counts =  new index_type[n];
//     std::fill_n(nnz_counts, n, 0);
//     index_type* coo_rows = A_coo->getRowData("cpu");
//     index_type* coo_cols = A_coo->getColData("cpu");
//     real_type* coo_vals = A_coo->getValues("cpu");

//     index_type* diag_control = new index_type[n]; //for DEDUPLICATION of the diagonal
//     std::fill_n(diag_control, n, 0);
//     index_type nnz_unpacked_no_duplicates = 0;
//     index_type nnz_no_duplicates = nnz;


//     //maybe check if they exist?
//     for (index_type i = 0; i < nnz; ++i)
//     {
//       nnz_counts[coo_rows[i]]++;
//       nnz_unpacked++;
//       nnz_unpacked_no_duplicates++;
//       if ((coo_rows[i] != coo_cols[i])&& (symmetric) && (!expanded))
//       {
//         nnz_counts[coo_cols[i]]++;
//         nnz_unpacked++;
//         nnz_unpacked_no_duplicates++;
//       }
//       if (coo_rows[i] == coo_cols[i]){
//         if (diag_control[coo_rows[i]] > 0) {
//          //duplicate
//           nnz_unpacked_no_duplicates--;
//           nnz_no_duplicates--;
//         }
//         diag_control[coo_rows[i]]++;
//       }
//     }
//     A_csr->setExpanded(true);
//     A_csr->setNnzExpanded(nnz_unpacked_no_duplicates);
//     index_type* csr_ia = new index_type[n+1];
//     std::fill_n(csr_ia, n + 1, 0);
//     index_type* csr_ja = new index_type[nnz_unpacked];
//     real_type* csr_a = new real_type[nnz_unpacked];
//     index_type* nnz_shifts = new index_type[n];
//     std::fill_n(nnz_shifts, n , 0);

//     IndexValuePair* tmp = new IndexValuePair[nnz_unpacked]; 

//     csr_ia[0] = 0;

//     for (index_type i = 1; i < n + 1; ++i){
//       csr_ia[i] = csr_ia[i - 1] + nnz_counts[i - 1] - (diag_control[i-1] - 1);
//     }

//     int r, start;


//     for (index_type i = 0; i < nnz; ++i){
//       //which row
//       r = coo_rows[i];
//       start = csr_ia[r];

//       if ((start + nnz_shifts[r]) > nnz_unpacked) {
//         out::warning() << "index out of bounds (case 1) start: " << start << "nnz_shifts[" << r << "] = " << nnz_shifts[r] << std::endl;
//       }
//       if ((r == coo_cols[i]) && (diag_control[r] > 1)) {//diagonal, and there are duplicates
//         bool already_there = false;  
//         for (index_type j = start; j < start + nnz_shifts[r]; ++j)
//         {
//           index_type c = tmp[j].getIdx();
//           if (c == r) {
//             real_type val = tmp[j].getValue();
//             val += coo_vals[i];
//             tmp[j].setValue(val);
//             already_there = true;
//             out::warning() << " duplicate found, row " << c << " adding in place " << j << " current value: " << val << std::endl;
//           }  
//         }  
//         if (!already_there){ // first time this duplicates appears

//           tmp[start + nnz_shifts[r]].setIdx(coo_cols[i]);
//           tmp[start + nnz_shifts[r]].setValue(coo_vals[i]);

//           nnz_shifts[r]++;
//         }
//       } else {//not diagonal
//         tmp[start + nnz_shifts[r]].setIdx(coo_cols[i]);
//         tmp[start + nnz_shifts[r]].setValue(coo_vals[i]);
//         nnz_shifts[r]++;

//         if ((coo_rows[i] != coo_cols[i]) && (symmetric == 1))
//         {
//           r = coo_cols[i];
//           start = csr_ia[r];

//           if ((start + nnz_shifts[r]) > nnz_unpacked)
//             out::warning() << "index out of bounds (case 2) start: " << start << "nnz_shifts[" << r << "] = " << nnz_shifts[r] << std::endl;
//           tmp[start + nnz_shifts[r]].setIdx(coo_rows[i]);
//           tmp[start + nnz_shifts[r]].setValue(coo_vals[i]);
//           nnz_shifts[r]++;
//         }
//       }
//     }
//     //now sort whatever is inside rows

//     for (int i = 0; i < n; ++i)
//     {

//       //now sorting (and adding 1)
//       int colStart = csr_ia[i];
//       int colEnd = csr_ia[i + 1];
//       int length = colEnd - colStart;
//       std::sort(&tmp[colStart],&tmp[colStart] + length);
//     }

//     for (index_type i = 0; i < nnz_unpacked; ++i)
//     {
//       csr_ja[i] = tmp[i].getIdx();
//       csr_a[i] = tmp[i].getValue();
//     }
// #if 0
//     for (int i = 0; i<n; ++i){
//       printf("Row: %d \n", i);
//       for (int j = csr_ia[i]; j<csr_ia[i+1]; ++j){
//         printf("(%d %16.16f) ", csr_ja[j], csr_a[j]);
//       }
//       printf("\n");
//     }
// #endif
//     A_csr->setNnz(nnz_no_duplicates);
//     if (memspace == "cpu"){
//       A_csr->updateData(csr_ia, csr_ja, csr_a, "cpu", "cpu");
//     } else {
//       if (memspace == "cuda"){      
//         A_csr->updateData(csr_ia, csr_ja, csr_a, "cpu", "cuda");
//       } else {
//         //display error
//       }
//     }
//     delete [] nnz_counts;
//     delete [] tmp;
//     delete [] nnz_shifts;
//     delete [] csr_ia;
//     delete [] csr_ja;
//     delete [] csr_a;
//     delete [] diag_control; 

//     return 0;
//   }

  int MatrixHandlerCpu::matvec(matrix::Sparse* Ageneric, 
                               vector_type* vec_x, 
                               vector_type* vec_result, 
                               const real_type* alpha, 
                               const real_type* beta,
                               std::string matrixFormat) 
  {
    using namespace constants;
    // int error_sum = 0;
    if (matrixFormat == "csr") {
      matrix::Csr* A = (matrix::Csr*) Ageneric;
      //result = alpha *A*x + beta * result
      index_type* ia = A->getRowData("cpu");
      index_type* ja = A->getColData("cpu");
      real_type*   a = A->getValues("cpu");

      real_type* x_data      = vec_x->getData("cpu");
      real_type* result_data = vec_result->getData("cpu");
      real_type sum;
      real_type y;
      real_type t;
      real_type c;

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
      out::error() << "MatVec not implemented (yet) for " 
                   << matrixFormat << " matrix format." << std::endl;
      return 1;
    }
  }

  int MatrixHandlerCpu::Matrix1Norm(matrix::Sparse* /* A */, real_type* /* norm */)
  {
    return -1;
  }

  /**
   * @authors Slaven Peles <peless@ornl.gov>, Daniel Reynolds (SMU), and
   * David Gardner and Carol Woodward (LLNL)
   */
  int MatrixHandlerCpu::csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr)
  {
    int error_sum = 0;
    assert(A_csc->getNnz() == A_csr->getNnz());
    assert(A_csc->getNumRows() == A_csr->getNumColumns());
    assert(A_csr->getNumRows() == A_csc->getNumColumns());

    index_type nnz = A_csc->getNnz();
    index_type n   = A_csc->getNumColumns();

    index_type* rowIdxCsc = A_csc->getRowData("cpu");
    index_type* colPtrCsc = A_csc->getColData("cpu");
    real_type*  valuesCsc = A_csc->getValues("cpu");

    index_type* rowPtrCsr = A_csr->getRowData("cpu");
    index_type* colIdxCsr = A_csr->getColData("cpu");
    real_type*  valuesCsr = A_csr->getValues("cpu");

    // Set all CSR row pointers to zero
    for (index_type i = 0; i <= n; ++i) {
      rowPtrCsr[i] = 0;
    }

    // Set all CSR values and column indices to zero
    for (index_type i = 0; i < nnz; ++i) {
      colIdxCsr[i] = 0;
      valuesCsr[i] = 0.0;
    }

    // Compute number of entries per row
    for (index_type i = 0; i < nnz; ++i) {
      rowPtrCsr[rowIdxCsc[i]]++;
    }

    // Compute cumualtive sum of nnz per row
    for (index_type row = 0, rowsum = 0; row < n; ++row)
    {
      // Store value in row pointer to temp
      index_type temp  = rowPtrCsr[row];

      // Copy cumulative sum to the row pointer
      rowPtrCsr[row] = rowsum;

      // Update row sum
      rowsum += temp;
    }
    rowPtrCsr[n] = nnz;

    for (index_type col = 0; col < n; ++col)
    {
      // Compute positions of column indices and values in CSR matrix and store them there
      // Overwrites CSR row pointers in the process
      for (index_type jj = colPtrCsc[col]; jj < colPtrCsc[col+1]; jj++)
      {
          index_type row  = rowIdxCsc[jj];
          index_type dest = rowPtrCsr[row];

          colIdxCsr[dest] = col;
          valuesCsr[dest] = valuesCsc[jj];

          rowPtrCsr[row]++;
      }
    }

    // Restore CSR row pointer values
    for (index_type row = 0, last = 0; row <= n; row++)
    {
        index_type temp  = rowPtrCsr[row];
        rowPtrCsr[row] = last;
        last    = temp;
    }

    return 0;
  }

} // namespace ReSolve
