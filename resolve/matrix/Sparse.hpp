// Matrix utilities
// Mirroring memory approach 
#pragma once
#include <string>
#include <resolve/Common.hpp>

namespace ReSolve { namespace matrix {
  class Sparse 
  {
    public:
      //basic constructor
      Sparse();
      Sparse(index_type n, index_type m, index_type nnz);
      Sparse(index_type n, 
             index_type m, 
             index_type nnz,
             bool symmetric,
             bool expanded);
      virtual ~Sparse();

      // accessors
      index_type getNumRows();
      index_type getNumColumns();
      index_type getNnz();
      index_type getNnzExpanded();

      bool symmetric(); 
      bool expanded();
      void setSymmetric(bool symmetric);
      void setExpanded(bool expanded);
      void setNnzExpanded(index_type nnz_expanded_new);
      void setNnz(index_type nnz_new); // for resetting when removing duplicates
      index_type setUpdated(std::string what);

      virtual index_type* getRowData(std::string memspace) = 0;
      virtual index_type* getColData(std::string memspace) = 0;
      virtual real_type* getValues(std::string memspace) = 0;

      virtual index_type updateData(index_type* row_data, index_type* col_data, real_type* val_data, std::string memspaceIn, std::string memspaceOut) = 0;
      virtual index_type updateData(index_type* row_data, index_type* col_data, real_type* val_data, index_type new_nnz, std::string memspaceIn, std::string memspaceOut) = 0;

      virtual index_type allocateMatrixData(std::string memspace) = 0;
      index_type setMatrixData(index_type* row_data, index_type* col_data, real_type* val_data, std::string memspace);

      index_type destroyMatrixData(std::string memspace);

      virtual void print() = 0;


      //update Values just updates values; it allocates if necessary.
      //values have the same dimensions between different formats 
      virtual int updateValues(real_type* new_vals, std::string memspaceIn, std::string memspaceOut);
      
      //set new values just sets the pointer, use caution.   
      virtual int setNewValues(real_type* new_vals, std::string memspace);
    
    protected:
      //size
      index_type n_{0};
      index_type m_{0};
      index_type nnz_{0};
      index_type nnz_expanded_{0};

      bool is_symmetric_{false};
      bool is_expanded_{false};

      //host data
      index_type* h_row_data_{nullptr};
      index_type* h_col_data_{nullptr};
      real_type* h_val_data_{nullptr};
      bool h_data_updated_{false};

      //gpu data
      index_type* d_row_data_{nullptr};
      index_type* d_col_data_{nullptr};
      real_type* d_val_data_{nullptr};
      bool d_data_updated_{false};

      void setNotUpdated();

  };
}} // namespace ReSolve::matrix
