// Matrix utilities
// Mirroring memory approach 
#pragma once
#include <string>
#include "resolveCommon.hpp"
#include<cstring>

namespace ReSolve 
{
  class resolveMatrix 
  {
    public:
      //basic constructor
      resolveMatrix();
      resolveMatrix(resolveInt n, resolveInt m, resolveInt nnz);
      resolveMatrix(resolveInt n, 
                    resolveInt m, 
                    resolveInt nnz,
                    bool symmetric,
                    bool expanded);
      ~resolveMatrix();

      // accessors
      resolveInt getNumRows();
      resolveInt getNumColumns();
      resolveInt getNnz();
      resolveInt getNnzExpanded();

      bool symmetric(); 
      bool expanded();
      void setSymmetric(bool symmetric);
      void setExpanded(bool expanded);
      void setNnzExpanded(resolveInt nnz_expanded_new);
      void setNnz(resolveInt nnz_new); // for resetting when removing duplicates
      void setUpdated(std::string what);

      resolveInt* getCsrRowPointers(std::string memspace);
      resolveInt* getCsrColIndices(std::string memspace);
      resolveReal* getCsrValues(std::string memspace);

      resolveInt* getCscColPointers(std::string memspace);
      resolveInt* getCscRowIndices(std::string memspace);
      resolveReal* getCscValues(std::string memspace);

      resolveInt* getCooRowIndices(std::string memspace);
      resolveInt* getCooColIndices(std::string memspace);
      resolveReal* getCooValues(std::string memspace);

      // Set functions just set the values. It is always a pointer copy, not a deep copy.
      resolveInt setCsr(resolveInt* csr_p, resolveInt* csr_i, resolveReal* csr_x, std::string memspace);
      resolveInt setCsc(resolveInt* csc_p, resolveInt* csc_i, resolveReal* csc_x, std::string memspace);
      resolveInt setCoo(resolveInt* coo_rows, resolveInt* coo_cols, resolveReal* coo_vals, std::string memspace);

      // Update functions update the data. There is always a deep copy, never a pointer copy
      // These function would allocate the space, if necessary.
      resolveInt updateCsr(resolveInt* csr_p, resolveInt* csr_i, resolveReal* csr_x, std::string memspaceIn, std::string memspaceOut);
      resolveInt updateCsc(resolveInt* csc_p, resolveInt* csc_i, resolveReal* csc_x, std::string memspaceIn, std::string memspaceOut);
      resolveInt updateCoo(resolveInt* coo_rows, resolveInt* coo_cols, resolveReal* coo_vals, std::string memspaceIn, std::string memspaceOut);

      //these functions should be used when, for instance, nnz changes (matrix get expanded, etc)

      resolveInt updateCsr(resolveInt* csr_p, resolveInt* csr_i, resolveReal* csr_x, resolveInt new_nnz, std::string memspaceIn, std::string memspaceOut);
      resolveInt updateCsc(resolveInt* csc_p, resolveInt* csc_i, resolveReal* csc_x, resolveInt new_nnz, std::string memspaceIn, std::string memspaceOut);
      resolveInt updateCoo(resolveInt* coo_rows, resolveInt* coo_cols, resolveReal* coo_vals, resolveInt new_nnz,  std::string memspaceIn, std::string memspaceOut);

      //DESTROY!
      resolveInt destroyCsr(std::string memspace);
      resolveInt destroyCsc(std::string memspace);
      resolveInt destroyCoo(std::string memspace);

      //allocate, sometimes needed
      void allocateCsr(std::string memspace);
      void allocateCsc(std::string memspace);
      void allocateCoo(std::string memspace);
 
    private:
      //size
      resolveInt n_;
      resolveInt m_;
      resolveInt nnz_;
      resolveInt nnz_expanded_;

      bool is_symmetric_;
      bool is_expanded_;

      //host data
      // COO format:
      resolveInt* h_coo_rows_;
      resolveInt* h_coo_cols_;
      resolveReal* h_coo_vals_;
      bool h_coo_updated_;

      // CSR format:
      resolveInt* h_csr_p_; //row starts
      resolveInt* h_csr_i_; //column indices
      resolveReal* h_csr_x_;//values 
      bool h_csr_updated_;

      // CSC format:
      resolveInt* h_csc_p_; //column starts
      resolveInt* h_csc_i_; //row indices
      resolveReal* h_csc_x_;//values 
      bool h_csc_updated_;

      //device data

      /* note -- COO format not typically kept on the gpu anyways */

      //COO format:
      resolveInt* d_coo_rows_;
      resolveInt* d_coo_cols_;
      resolveReal* d_coo_vals_;
      bool d_coo_updated_;

      // CSR format:
      resolveInt* d_csr_p_; //row starts
      resolveInt* d_csr_i_; //column indices
      resolveReal* d_csr_x_;//values 
      bool d_csr_updated_;

      // CSC format:
      resolveInt* d_csc_p_; //column starts
      resolveInt* d_csc_i_; //row indices
      resolveReal* d_csc_x_;//values  
      bool d_csc_updated_;

      //auxiliary functions for managing updating data between cpu and cuda
      void setNotUpdated();
      void copyCsr(std::string memspaceOut);
      void copyCsc(std::string memspaceOut);
      void copyCoo(std::string memspaceOut);
  };
}
