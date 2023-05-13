// Matrix utilities
// Mirroring memory approach 
#pragma once
#include <string>
#include "resolveCommon.hpp"
#include<cstring>
namespace ReSolve {
  class resolveMatrix {

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

      
      resolveInt* getCsrRowPointers(std::string memspace);
      resolveInt* getCsrColIndices(std::string memspace);
      resolveReal* getCsrValues(std::string memspace);

      resolveInt* getCscColPointers(std::string memspace);
      resolveInt* getCscRowIndices(std::string memspace);
      resolveReal* getCscValues(std::string memspace);

      resolveInt* getCooRowIndices(std::string memspace);
      resolveInt* getCooColIndices(std::string memspace);
      resolveReal* getCooValues(std::string memspace);
// n does not change but nnz might!
      resolveInt setCsr(resolveInt* csr_p, resolveInt* csr_i, resolveReal* csr_x, resolveInt new_nnz, std::string memspaceIn, std::string memspaceOut);
      resolveInt setCsc(resolveInt* csc_p, resolveInt* csc_i, resolveReal* csc_x, resolveInt new_nnz, std::string memspace);
      resolveInt setCoo(resolveInt* coo_rows, resolveInt* coo_cols, resolveReal* coo_vals, resolveInt new_nnz, std::string memspace);



//DESTROY!
      resolveInt destroyCsr(std::string memspace);
      resolveInt destroyCsc(std::string memspace);
      resolveInt destroyCoo(std::string memspace);
       

    private:
      //size
      resolveInt n;
      resolveInt m;
      resolveInt nnz;
      resolveInt nnz_expanded;

      bool is_symmetric;
      bool is_expanded;
      
      //host data
      // COO format:
      resolveInt* h_coo_rows;
      resolveInt* h_coo_cols;
      resolveReal* h_coo_vals;

      // CSR format:
      resolveInt* h_csr_p; //row starts
      resolveInt* h_csr_i; //column indices
      resolveReal* h_csr_x;//values 

      // CSC format:
      resolveInt* h_csc_p; //column starts
      resolveInt* h_csc_i; //row indices
      resolveReal* h_csc_x;//values 

      //device data

      /* note -- COO format not typically kept on the gpu anyways */

      //COO format:
      resolveInt* d_coo_rows;
      resolveInt* d_coo_cols;
      resolveReal* d_coo_vals;

      // CSR format:
      resolveInt* d_csr_p; //row starts
      resolveInt* d_csr_i; //column indices
      resolveReal* d_csr_x;//values 

      // CSC format:
      resolveInt* d_csc_p; //column starts
      resolveInt* d_csc_i; //row indices
      resolveReal* d_csc_x;//values 
      

  };
}
