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
      Int setUpdated(std::string what);

      virtual Int* getRowData(std::string memspace){return nullptr;};
      virtual Int* getColData(std::string memspace){return nullptr;};
      virtual Real* getValues(std::string memspace){return nullptr;}; 

      virtual Int updateData(Int* row_data, Int* col_data, Real* val_data, std::string memspaceIn, std::string memspaceOut){return -1;}; 
      virtual Int updateData(Int* row_data, Int* col_data, Real* val_data, Int new_nnz, std::string memspaceIn, std::string memspaceOut){return -1;}; 

      virtual Int allocateMatrixData(std::string memspace){return -1;}; 
      Int setMatrixData(Int* row_data, Int* col_data, Real* val_data, std::string memspace);

      Int destroyMatrixData(std::string memspace);
    
    protected:
      //size
      Int n_;
      Int m_;
      Int nnz_;
      Int nnz_expanded_;

      bool is_symmetric_;
      bool is_expanded_;

      //host data
      Int* h_row_data_;
      Int* h_col_data_;
      Real* h_val_data_;

      bool h_data_updated_;

      //gpu data
      Int* d_row_data_;
      Int* d_col_data_;
      Real* d_val_data_;
      bool d_data_updated_;

      void setNotUpdated();

  };
}
