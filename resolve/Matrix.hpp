// Matrix utilities
// Mirroring memory approach 
#pragma once
#include <string>
#include "Common.hpp"

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
      virtual ~Matrix();

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

      virtual Int* getRowData(std::string memspace) = 0;
      virtual Int* getColData(std::string memspace) = 0;
      virtual Real* getValues(std::string memspace) = 0;

      virtual Int updateData(Int* row_data, Int* col_data, Real* val_data, std::string memspaceIn, std::string memspaceOut) = 0;
      virtual Int updateData(Int* row_data, Int* col_data, Real* val_data, Int new_nnz, std::string memspaceIn, std::string memspaceOut) = 0;

      virtual Int allocateMatrixData(std::string memspace) = 0;
      Int setMatrixData(Int* row_data, Int* col_data, Real* val_data, std::string memspace);

      Int destroyMatrixData(std::string memspace);

      virtual void print() = 0;
    
    protected:
      //size
      Int n_{0};
      Int m_{0};
      Int nnz_{0};
      Int nnz_expanded_{0};

      bool is_symmetric_{false};
      bool is_expanded_{false};

      //host data
      Int* h_row_data_{nullptr};
      Int* h_col_data_{nullptr};
      Real* h_val_data_{nullptr};
      bool h_data_updated_{false};

      //gpu data
      Int* d_row_data_{nullptr};
      Int* d_col_data_{nullptr};
      Real* d_val_data_{nullptr};
      bool d_data_updated_{false};

      void setNotUpdated();

  };
}
