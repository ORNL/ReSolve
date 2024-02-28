#include <math.h>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Csr.hpp>
#include "LinSolverDirectSerialILU0.hpp"

#include <resolve/utilities/logger/Logger.hpp>

namespace ReSolve 
{
  LinSolverDirectSerialILU0::LinSolverDirectSerialILU0(LinAlgWorkspaceCpu* workspace)
  {
    workspace_ = workspace;
  }

  LinSolverDirectSerialILU0::~LinSolverDirectSerialILU0()
  {
    if (owns_factors_) {
      delete L_;
      delete U_;
      L_ = nullptr;
      U_ = nullptr;
    }
    delete [] h_aux1_;
    delete [] h_ILU_vals_;
  }

  int LinSolverDirectSerialILU0::setup(matrix::Sparse* A,
                                       matrix::Sparse*,
                                       matrix::Sparse*,
                                       index_type*,
                                       index_type*,
                                       vector_type* )
  {
    this->A_ = (matrix::Csr*) A;
    index_type n = A_->getNumRows();
    index_type nnz = A_->getNnzExpanded();

    h_ILU_vals_ = new real_type[nnz];
    h_aux1_ = new real_type[n];

    index_type zero_pivot = 0; // no zero pivot 

    //copy A values to a buffer first
    for (index_type i = 0; i < nnz; ++i) {
      h_ILU_vals_[i] = A_->getValues(ReSolve::memory::HOST)[i];
    }

    // allocate helper variables that make this code fast
    index_type* u_ptr     = new index_type[n];
    index_type* ja_mapper = new index_type[n];

    // aux scalars for indexing etc
    index_type k, j,  jw, j1, j2;    

    for (index_type i = 0; i < n; ++i) {
      j1 = A_->getRowData(ReSolve::memory::HOST)[i];
      j2 = A_->getRowData(ReSolve::memory::HOST)[i + 1];
      for (index_type j = j1; j < j2; ++j) {
        ja_mapper[A_->getColData(ReSolve::memory::HOST)[j]] = j;
      }
      // IJK ILU 
      j = j1;
      while ( j < j2) {
        k = A_->getColData(ReSolve::memory::HOST)[j];
        if (k < i) {
          h_ILU_vals_[j] /= h_ILU_vals_[u_ptr[k]];
          for (index_type jj = u_ptr[k] + 1; jj < A_->getRowData(ReSolve::memory::HOST)[k + 1]; ++jj) {
            jw = ja_mapper[A_->getColData(ReSolve::memory::HOST)[jj]];
            if (jw != 0) {
              h_ILU_vals_[jw] -= h_ILU_vals_[j] * h_ILU_vals_[jj];   
            }
          }   
        } else {
          break;
        }
        j++;    
      }
      u_ptr[i] = j;
      if ((k != i) || (fabs(h_ILU_vals_[j]) < 1e-16)) {
        zero_pivot = -1; // zero pivot is in place (i,i) on the diagonal
        return zero_pivot;
      }
      // reset mapper
      for (index_type j = j1; j< j2; ++j) {
        ja_mapper[A_->getColData(ReSolve::memory::HOST)[j]] = 0;
      }
    }

    //clean up
    delete [] ja_mapper;
    delete [] u_ptr;

    // split into L and U!
    index_type nnzL = 0, nnzU = 0;
    // the diagonal values GO TO U, L has 1s on the diagonal
    for (index_type i = 0; i < n; ++i) {
      j1 = A_->getRowData(ReSolve::memory::HOST)[i];
      j2 = A_->getRowData(ReSolve::memory::HOST)[i + 1];
      for (index_type j = j1; j < j2; ++j) {
        if (A->getColData(ReSolve::memory::HOST)[j] == i) {
          // diagonal, add to both
          nnzL++;
          nnzU++;
        }
        if (A->getColData(ReSolve::memory::HOST)[j] > i) {
          // upper part
          nnzU++;
        }
        if (A->getColData(ReSolve::memory::HOST)[j] < i) {
          // lower part
          nnzL++;
        }
      }  
    }
    // TODO: What is the purpose of nnzL and nnzU if they are not used after this?
    // allocate L and U

    L_ = new matrix::Csr(n, n, nnzL, false, true);
    U_ = new matrix::Csr(n, n, nnzU, false, true);
    owns_factors_ = true;

    L_->allocateMatrixData(ReSolve::memory::HOST);  
    U_->allocateMatrixData(ReSolve::memory::HOST);  
    index_type lit = 0, uit = 0, kL, kU; 
    L_->getRowData(ReSolve::memory::HOST)[0] = 0;
    U_->getRowData(ReSolve::memory::HOST)[0] = 0;

    for (index_type i = 0; i < n; ++i) {
      j1 = A_->getRowData(ReSolve::memory::HOST)[i];
      j2 = A_->getRowData(ReSolve::memory::HOST)[i + 1];
      kL = 0;
      kU = 0;
      for (index_type j = j1; j < j2; ++j) {

        if (A->getColData(ReSolve::memory::HOST)[j] == i) {
          // diagonal, add to both

          L_->getValues(ReSolve::memory::HOST)[lit] = 1.0;
          U_->getValues(ReSolve::memory::HOST)[uit] = h_ILU_vals_[j];

          L_->getColData(ReSolve::memory::HOST)[lit] = i;
          U_->getColData(ReSolve::memory::HOST)[uit] = i;

          lit++;
          uit++;
          kL++;
          kU++;
        }

        if (A->getColData(ReSolve::memory::HOST)[j] > i) {
          // upper part

          U_->getValues(ReSolve::memory::HOST) [uit] = h_ILU_vals_[j]; 
          U_->getColData(ReSolve::memory::HOST)[uit] = A_->getColData(ReSolve::memory::HOST)[j]; ;

          uit++;
          kU++;
        }

        if (A->getColData(ReSolve::memory::HOST)[j] < i) {
          // lower part
          L_->getValues(ReSolve::memory::HOST) [lit] =  h_ILU_vals_[j]; 
          L_->getColData(ReSolve::memory::HOST)[lit] = A_->getColData(ReSolve::memory::HOST)[j]; 

          lit++;
          kL++;
        }
      }  
      //update row pointers
      L_->getRowData(ReSolve::memory::HOST)[i + 1] = L_->getRowData(ReSolve::memory::HOST)[i] + kL; 
      U_->getRowData(ReSolve::memory::HOST)[i + 1] = U_->getRowData(ReSolve::memory::HOST)[i] + kU; 
    }
   
    return zero_pivot;
  }

  int LinSolverDirectSerialILU0::reset(matrix::Sparse* A)
  {
    return this->setup(A);
  }
  // solution is returned in RHS
  int LinSolverDirectSerialILU0::solve(vector_type* rhs)
  {
    int error_sum = 0;
    // printf("solve t 1\n");
    // h_aux1 = L^{-1} rhs
    for (index_type i = 0; i < L_->getNumRows(); ++i) {
      h_aux1_[i] = rhs->getData(ReSolve::memory::HOST)[i];
      for (index_type j = L_->getRowData(ReSolve::memory::HOST)[i]; j < L_->getRowData(ReSolve::memory::HOST)[i + 1] - 1; ++j) {
        index_type col = L_->getColData(ReSolve::memory::HOST)[j];
        h_aux1_[i] -= L_->getValues(ReSolve::memory::HOST)[j] * h_aux1_[col]; 
      }
      h_aux1_[i] /= L_->getValues(ReSolve::memory::HOST)[L_->getRowData(ReSolve::memory::HOST)[i + 1] - 1];
    }

    // rhs = U^{-1} h_aux1

    for (index_type i = A_->getNumRows() - 1; i >= 0; --i) {
      rhs->getData(ReSolve::memory::HOST)[i] = h_aux1_[i];
      for (index_type j = U_->getRowData(ReSolve::memory::HOST)[i] + 1; j < U_->getRowData(ReSolve::memory::HOST)[i + 1]; ++j) {
        index_type col = U_->getColData(ReSolve::memory::HOST)[j];
        rhs->getData(ReSolve::memory::HOST)[i] -= U_->getValues(ReSolve::memory::HOST)[j] * rhs->getData(ReSolve::memory::HOST)[col];
      }
      rhs->getData(ReSolve::memory::HOST)[i] /= U_->getValues(ReSolve::memory::HOST)[U_->getRowData(ReSolve::memory::HOST)[i]]; //divide by the diagonal entry
    }

    return error_sum;
  }

  int LinSolverDirectSerialILU0::solve(vector_type* rhs, vector_type* x)
  {
    //printf("solve t 2i, L has %d rows, U has %d rows \n", L_->getNumRows(), U_->getNumRows());
    int error_sum = 0;
    // h_aux1 = L^{-1} rhs
      //for (int ii=0; ii<10; ++ii) printf("y[%d] = %16.16f \n ", ii,   rhs->getData(ReSolve::memory::HOST)[ii]); 
    for (index_type i = 0; i < L_->getNumRows(); ++i) {
      h_aux1_[i] = rhs->getData(ReSolve::memory::HOST)[i];
      for (index_type j = L_->getRowData(ReSolve::memory::HOST)[i]; j < L_->getRowData(ReSolve::memory::HOST)[i + 1] - 1; ++j) {
        index_type col = L_->getColData(ReSolve::memory::HOST)[j];
        h_aux1_[i] -= L_->getValues(ReSolve::memory::HOST)[j] * h_aux1_[col]; 
      }
      h_aux1_[i] /= L_->getValues(ReSolve::memory::HOST)[L_->getRowData(ReSolve::memory::HOST)[i + 1] - 1];
    }

    //for (int ii=0; ii<10; ++ii) printf("(L)^{-1}y[%d] = %16.16f \n ", ii,  h_aux1_[ii]); 
    // x = U^{-1} h_aux1

    for (index_type i = U_->getNumRows() - 1; i >= 0; --i) {
      x->getData(ReSolve::memory::HOST)[i] = h_aux1_[i];
      for (index_type j = U_->getRowData(ReSolve::memory::HOST)[i] + 1; j < U_->getRowData(ReSolve::memory::HOST)[i + 1]; ++j) {
        index_type col = U_->getColData(ReSolve::memory::HOST)[j];
        x->getData(ReSolve::memory::HOST)[i] -= U_->getValues(ReSolve::memory::HOST)[j] *  x->getData(ReSolve::memory::HOST)[col];
      }
      x->getData(ReSolve::memory::HOST)[i] /= U_->getValues(ReSolve::memory::HOST)[U_->getRowData(ReSolve::memory::HOST)[i]]; //divide by the diagonal entry
    }
    //for (int ii=0; ii<10; ++ii) printf("(LU)^{-1}y[%d] = %16.16f \n ", ii,  x->getData(ReSolve::memory::HOST)[ii]); 
   return error_sum;
  }
} // namespace resolve
