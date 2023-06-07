// Matrix utilities
// Mirroring memory approach 
#pragma once
#include <string>
#include "Common.hpp"
#include<cstring>

namespace ReSolve 
{
  class Matrix 
  {
    public:
      //basic constructor
      Matrix();
      Matrix(Int n, Int m, Int nnz);
      Matrix(Int n, 
                    Int m, 
                    Int nnz,
                    bool symmetric,
                    bool expanded);
      ~Matrix();

      // accessors
      Int getNumRows();
      Int getNumColumns();
      Int getNnz();
      Int getNnzExpanded();

      bool symmetric(); 
      bool expanded();
      void setSymmetric(bool symmetric);
      void setExpanded(bool expanded);
      void setNnzExpanded(Int nnz_expanded_new);
      void setNnz(Int nnz_new); // for resetting when removing duplicates
      void setUpdated(std::string what);

      Int* getCsrRowPointers(std::string memspace);
      Int* getCsrColIndices(std::string memspace);
      Real* getCsrValues(std::string memspace);

      Int* getCscColPointers(std::string memspace);
      Int* getCscRowIndices(std::string memspace);
      Real* getCscValues(std::string memspace);

      Int* getCooRowIndices(std::string memspace);
      Int* getCooColIndices(std::string memspace);
      Real* getCooValues(std::string memspace);

      // Set functions just set the values. It is always a pointer copy, not a deep copy.
      Int setCsr(Int* csr_p, Int* csr_i, Real* csr_x, std::string memspace);
      Int setCsc(Int* csc_p, Int* csc_i, Real* csc_x, std::string memspace);
      Int setCoo(Int* coo_rows, Int* coo_cols, Real* coo_vals, std::string memspace);

      // Update functions update the data. There is always a deep copy, never a pointer copy
      // These function would allocate the space, if necessary.
      Int updateCsr(Int* csr_p, Int* csr_i, Real* csr_x, std::string memspaceIn, std::string memspaceOut);
      Int updateCsc(Int* csc_p, Int* csc_i, Real* csc_x, std::string memspaceIn, std::string memspaceOut);
      Int updateCoo(Int* coo_rows, Int* coo_cols, Real* coo_vals, std::string memspaceIn, std::string memspaceOut);

      //these functions should be used when, for instance, nnz changes (matrix get expanded, etc)

      Int updateCsr(Int* csr_p, Int* csr_i, Real* csr_x, Int new_nnz, std::string memspaceIn, std::string memspaceOut);
      Int updateCsc(Int* csc_p, Int* csc_i, Real* csc_x, Int new_nnz, std::string memspaceIn, std::string memspaceOut);
      Int updateCoo(Int* coo_rows, Int* coo_cols, Real* coo_vals, Int new_nnz,  std::string memspaceIn, std::string memspaceOut);

      //DESTROY!
      Int destroyCsr(std::string memspace);
      Int destroyCsc(std::string memspace);
      Int destroyCoo(std::string memspace);

      //allocate, sometimes needed
      void allocateCsr(std::string memspace);
      void allocateCsc(std::string memspace);
      void allocateCoo(std::string memspace);
 
    private:
      //size
      Int n_;
      Int m_;
      Int nnz_;
      Int nnz_expanded_;

      bool is_symmetric_;
      bool is_expanded_;

      //host data
      // COO format:
      Int* h_coo_rows_;
      Int* h_coo_cols_;
      Real* h_coo_vals_;
      bool h_coo_updated_;

      // CSR format:
      Int* h_csr_p_; //row starts
      Int* h_csr_i_; //column indices
      Real* h_csr_x_;//values 
      bool h_csr_updated_;

      // CSC format:
      Int* h_csc_p_; //column starts
      Int* h_csc_i_; //row indices
      Real* h_csc_x_;//values 
      bool h_csc_updated_;

      //device data

      /* note -- COO format not typically kept on the gpu anyways */

      //COO format:
      Int* d_coo_rows_;
      Int* d_coo_cols_;
      Real* d_coo_vals_;
      bool d_coo_updated_;

      // CSR format:
      Int* d_csr_p_; //row starts
      Int* d_csr_i_; //column indices
      Real* d_csr_x_;//values 
      bool d_csr_updated_;

      // CSC format:
      Int* d_csc_p_; //column starts
      Int* d_csc_i_; //row indices
      Real* d_csc_x_;//values  
      bool d_csc_updated_;

      //auxiliary functions for managing updating data between cpu and cuda
      void setNotUpdated();
      void copyCsr(std::string memspaceOut);
      void copyCsc(std::string memspaceOut);
      void copyCoo(std::string memspaceOut);
  };
}
