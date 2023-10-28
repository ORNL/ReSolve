#include <algorithm>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/utilities/misc/IndexValuePair.hpp>
#include "MatrixHandler.hpp"
#include "MatrixHandlerCpu.hpp"

#ifdef RESOLVE_USE_CUDA
#include "MatrixHandlerCuda.hpp"
#endif
#ifdef RESOLVE_USE_HIP
#include "MatrixHandlerHip.hpp"
#endif

namespace ReSolve {
  // Create a shortcut name for Logger static class
  using out = io::Logger;

  /**
   * @brief Default constructor
   * 
   * @post Instantiates CPU and CUDA matrix handlers, but does not 
   * create a workspace.
   * 
   * @todo There is little utility for the default constructor. Rethink its purpose.
   * Consider making it private method.
   */
  MatrixHandler::MatrixHandler()
  {
    new_matrix_ = true;
    cpuImpl_    = new MatrixHandlerCpu();
  }

  /**
   * @brief Destructor
   * 
   */
  MatrixHandler::~MatrixHandler()
  {
    if (isCpuEnabled_)  delete cpuImpl_;
    if (isCudaEnabled_) delete cudaImpl_;
    if (isHipEnabled_) delete hipImpl_;
  }

  /**
   * @brief Constructor taking pointer to the workspace as its parameter.
   * 
   * @note The CPU implementation currently does not require a workspace.
   * The workspace pointer parameter is provided for forward compatibility.
   */
  MatrixHandler::MatrixHandler(LinAlgWorkspaceCpu* new_workspace)
  {
    cpuImpl_  = new MatrixHandlerCpu(new_workspace);
    isCpuEnabled_  = true;
    isCudaEnabled_ = false;
  }

#ifdef RESOLVE_USE_CUDA
  /**
   * @brief Constructor taking pointer to the CUDA workspace as its parameter.
   * 
   * @post A CPU implementation instance is created because it is cheap and
   * it does not require a workspace.
   * 
   * @post A CUDA implementation instance is created with supplied workspace.
   */
  MatrixHandler::MatrixHandler(LinAlgWorkspaceCUDA* new_workspace)
  {
    cpuImpl_  = new MatrixHandlerCpu();
    cudaImpl_ = new MatrixHandlerCuda(new_workspace);
    isCpuEnabled_  = true;
    isCudaEnabled_ = true;
  }
#endif

#ifdef RESOLVE_USE_HIP
  /**
   * @brief Constructor taking pointer to the CUDA workspace as its parameter.
   * 
   * @post A CPU implementation instance is created because it is cheap and
   * it does not require a workspace.
   * 
   * @post A HIP implementation instance is created with supplied workspace.
   */
  MatrixHandler::MatrixHandler(LinAlgWorkspaceHIP* new_workspace)
  {
    cpuImpl_  = new MatrixHandlerCpu();
    hipImpl_ = new MatrixHandlerHip(new_workspace);
    isCpuEnabled_  = true;
    isHipEnabled_ = true;
  }
#endif
  void MatrixHandler::setValuesChanged(bool isValuesChanged, std::string memspace)
  {
    if (memspace == "cpu") {
      cpuImpl_->setValuesChanged(isValuesChanged);
    } else if (memspace == "cuda") {
      cudaImpl_->setValuesChanged(isValuesChanged);
    } else if (memspace == "hip") {
      hipImpl_->setValuesChanged(isValuesChanged);
    } else {
      out::error() << "Unsupported device " << memspace << "\n";
    }
  }

  /**
   * @brief Converts COO to CSR matrix format.
   * 
   * Conversion takes place on CPU, and then CSR matrix is copied to `memspace`.
   */
  int MatrixHandler::coo2csr(matrix::Coo* A_coo, matrix::Csr* A_csr, std::string memspace)
  {
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

    IndexValuePair* tmp = new IndexValuePair[nnz_unpacked]; 

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
      } else if (memspace == "hip"){      
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
   * @brief Matrix vector product: result = alpha * A * x + beta * result
   * 
   * @param[in]  A - Sparse matrix
   * @param[in]  vec_x - Vector multiplied by the matrix
   * @param[out] vec_result - Vector where the result is stored
   * @param[in]  alpha - scalar parameter
   * @param[in]  beta  - scalar parameter
   * @param[in]  matrixFormat - Only CSR format is supported at this time
   * @param[in]  memspace     - Device where the product is computed
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
    } else if (memspace == "hip") {
      printf("about to run mv");
        return hipImpl_->matvec(A, vec_x, vec_result, alpha, beta, matrixFormat);
    } else {
        out::error() << "Support for device " << memspace << " not implemented (yet)" << std::endl;
        return 1;
    }
  }


  int MatrixHandler::csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr, std::string memspace)
  {
    if (memspace == "cuda") { 
      return cudaImpl_->csc2csr(A_csc, A_csr);
    } else if (memspace == "hip") {
      return hipImpl_->csc2csr(A_csc, A_csr);
    } else if (memspace == "cpu") { 
      out::warning() << "Using untested csc2csr on CPU ..." << std::endl;
      return cpuImpl_->csc2csr(A_csc, A_csr);
    } else {
      out::error() << "csc2csr not implemented for " << memspace << " device." << std::endl;
      return -1;
    }
  }

} // namespace ReSolve
