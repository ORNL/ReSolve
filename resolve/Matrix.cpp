#include "Matrix.hpp"
#include <cuda_runtime.h>

namespace ReSolve 
{
  Matrix::Matrix()
  {
  }

  Matrix::Matrix(index_type n, 
                 index_type m, 
                 index_type nnz):
    n_{n},
    m_{m},
    nnz_{nnz}
  {
    this->is_symmetric_ = false;
    this->is_expanded_ = true;//defaults is a normal non-symmetric fully expanded matrix
    this->nnz_expanded_ = nnz;

    setNotUpdated();

    //set everything to nullptr
    h_row_data_ = nullptr;
    h_col_data_ = nullptr;
    h_val_data_ = nullptr;

    d_row_data_ = nullptr;
    d_col_data_ = nullptr;
    d_val_data_ = nullptr;
  }

  Matrix::Matrix(index_type n, 
                 index_type m, 
                 index_type nnz,
                 bool symmetric,
                 bool expanded):
    n_{n},
    m_{m},
    nnz_{nnz},
    is_symmetric_{symmetric},
    is_expanded_{expanded}
  {
    if (is_expanded_) {
      this->nnz_expanded_ = nnz_;
    } else {
      this->nnz_expanded_ = 0;
    }
    setNotUpdated();

    //set everything to nullptr
    h_row_data_ = nullptr;
    h_col_data_ = nullptr;
    h_val_data_ = nullptr;

    d_row_data_ = nullptr;
    d_col_data_ = nullptr;
    d_val_data_ = nullptr;
  }

  Matrix::~Matrix()
  {
    this->destroyMatrixData("cpu");
    this->destroyMatrixData("cuda");
  }

  void Matrix::setNotUpdated()
  {
    h_data_updated_ = false;
    d_data_updated_ = false; 
  }
  
index_type Matrix::getNumRows()
  {
    return this->n_;
  }

  index_type Matrix::getNumColumns()
  {
    return this->m_;
  }

  index_type Matrix::getNnz()
  {
    return this->nnz_;
  }

  index_type Matrix::getNnzExpanded()
  {
    return this->nnz_expanded_;
  }

  bool Matrix::symmetric()
  {
    return is_symmetric_;
  }

  bool Matrix::expanded()
  {
    return is_expanded_;
  }

  void Matrix::setSymmetric(bool symmetric)
  {
    this->is_symmetric_ = symmetric;
  }

  void Matrix::setExpanded(bool expanded)
  {
    this->is_expanded_ = expanded;
  }

  void Matrix::setNnzExpanded(index_type nnz_expanded_new)
  {
    this->nnz_expanded_ = nnz_expanded_new;
  }

  void Matrix::setNnz(index_type nnz_new)
  {
    this->nnz_ = nnz_new;
  }

  int Matrix::setUpdated(std::string what)
  {
    if (what == "cpu")
    {
      h_data_updated_ = true;
      d_data_updated_ = false;
    } else {
      if (what == "cuda"){
        d_data_updated_ = true;
        h_data_updated_ = false;
      } else {
        return -1;
      }
    }
    return 0;
  }

  int Matrix::setMatrixData(index_type* row_data, index_type* col_data, real_type* val_data, std::string memspace)
  {

    setNotUpdated();

    if (memspace == "cpu"){
      this->h_row_data_ = row_data;
      this->h_col_data_ = col_data;
      this->h_val_data_ = val_data;	
      h_data_updated_ = true;
    } else {
      if (memspace == "cuda"){ 
        this->d_row_data_ = row_data;
        this->d_col_data_ = col_data;
        this->d_val_data_ = val_data;	
        d_data_updated_ = true;
      } else {
        return -1;
      }
    }
    return 0;
  }

  index_type Matrix::destroyMatrixData(std::string memspace)
  { 
    if (memspace == "cpu"){  
      if (h_row_data_ != nullptr) delete [] h_row_data_;
      if (h_col_data_ != nullptr) delete [] h_col_data_;
      if (h_val_data_ != nullptr) delete [] h_val_data_;
    } else {
      if (memspace == "cuda"){ 
        if (d_row_data_ != nullptr) cudaFree(d_row_data_);
        if (d_col_data_ != nullptr) cudaFree(d_col_data_);
        if (d_val_data_ != nullptr) cudaFree(d_val_data_);
      } else {
        return -1;
      }
    }
    return 0;
  }
}
