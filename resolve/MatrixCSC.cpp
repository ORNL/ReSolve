#include <cstring>  // <-- includes memcpy
#include <cuda_runtime.h>
#include "MatrixCSC.hpp"

namespace ReSolve 
{
  MatrixCSC::MatrixCSC()
  {
  }

  MatrixCSC::MatrixCSC(index_type n, index_type m, index_type nnz) : Matrix(n, m, nnz)
  {
  }
  
  MatrixCSC::MatrixCSC(index_type n, 
                       index_type m, 
                       index_type nnz,
                       bool symmetric,
                       bool expanded) : Matrix(n, m, nnz, symmetric, expanded)
  {
  }

  MatrixCSC::~MatrixCSC()
  {
  }

  index_type* MatrixCSC::getRowData(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCsc("cpu");
      return this->h_row_data_;
    } else {
      if (memspace == "cuda") {
        copyCsc("cuda");
        return this->d_row_data_;
      } else {
        return nullptr;
      }
    }
  }

  index_type* MatrixCSC::getColData(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCsc("cpu");
      return this->h_col_data_;
    } else {
      if (memspace == "cuda") {
        copyCsc("cuda");
        return this->d_col_data_;
      } else {
        return nullptr;
      }
    }
  }

  real_type* MatrixCSC::getValues(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCsc("cpu");
      return this->h_val_data_;
    } else {
      if (memspace == "cuda") {
        copyCsc("cuda");
        return this->d_val_data_;
      } else {
        return nullptr;
      }
    }
  }

  index_type MatrixCSC::updateData(index_type* row_data, index_type* col_data, real_type* val_data, std::string memspaceIn, std::string memspaceOut)
  {
    index_type nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    //four cases (for now)
    int control=-1;
    setNotUpdated();
    if ((memspaceIn == "cpu") && (memspaceOut == "cpu")){ control = 0;}
    if ((memspaceIn == "cpu") && (memspaceOut == "cuda")){ control = 1;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cpu")){ control = 2;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cuda")){ control = 3;}

    if (memspaceOut == "cpu") {
      //check if cpu data allocated
      if (h_col_data_ == nullptr) {
        this->h_col_data_ = new index_type[n_ + 1];
      }
      if (h_row_data_ == nullptr) {
        this->h_row_data_ = new index_type[nnz_current];
      } 
      if (h_val_data_ == nullptr) {
        this->h_val_data_ = new real_type[nnz_current];
      }
    }

    if (memspaceOut == "cuda") {
      //check if cuda data allocated
      if (d_col_data_ == nullptr) {
        cudaMalloc(&d_col_data_, (n_ + 1) * sizeof(index_type)); 
      }
      if (d_row_data_ == nullptr) {
        cudaMalloc(&d_row_data_, nnz_current * sizeof(index_type)); 
      }
      if (d_val_data_ == nullptr) {
        cudaMalloc(&d_val_data_, nnz_current * sizeof(real_type)); 
      }
    }

    switch(control)  {
      case 0: //cpu->cpu
        std::memcpy(h_col_data_, col_data, (n_ + 1) * sizeof(index_type));
        std::memcpy(h_row_data_, row_data, (nnz_current) * sizeof(index_type));
        std::memcpy(h_val_data_, val_data, (nnz_current) * sizeof(real_type));
        h_data_updated_ = true;
        break;
      case 2://cuda->cpu
        cudaMemcpy(h_col_data_, col_data, (n_ + 1) * sizeof(index_type), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_row_data_, row_data, (nnz_current) * sizeof(index_type), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_val_data_, val_data, (nnz_current) * sizeof(real_type), cudaMemcpyDeviceToHost);
        h_data_updated_ = true;
        break;
      case 1://cpu->cuda
        cudaMemcpy(d_col_data_, col_data, (n_ + 1) * sizeof(index_type), cudaMemcpyHostToDevice);
        cudaMemcpy(d_row_data_, row_data, (nnz_current) * sizeof(index_type), cudaMemcpyHostToDevice);
        cudaMemcpy(d_val_data_, val_data, (nnz_current) * sizeof(real_type), cudaMemcpyHostToDevice);
        d_data_updated_ = true;
        break;
      case 3://cuda->cuda
        cudaMemcpy(d_col_data_, col_data, (n_ + 1) * sizeof(index_type), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_row_data_, row_data, (nnz_current) * sizeof(index_type), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_val_data_, val_data, (nnz_current) * sizeof(real_type), cudaMemcpyDeviceToDevice);
        d_data_updated_ = true;
        break;
      default:
        return -1;
    }
    return 0;

  } 

  index_type MatrixCSC::updateData(index_type* row_data, index_type* col_data, real_type* val_data, index_type new_nnz, std::string memspaceIn, std::string memspaceOut)
  {
    this->destroyMatrixData(memspaceOut);
    int i = this->updateData(col_data, row_data, val_data, memspaceIn, memspaceOut);
    return i;
  } 

  index_type MatrixCSC::allocateMatrixData(std::string memspace)
  {
    index_type nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    destroyMatrixData(memspace);//just in case

    if (memspace == "cpu") {
      this->h_col_data_ = new index_type[n_ + 1];
      this->h_row_data_ = new index_type[nnz_current];
      this->h_val_data_ = new real_type[nnz_current];
      return 0;
    }

    if (memspace == "cuda") {
      cudaMalloc(&d_col_data_, (n_ + 1) * sizeof(index_type)); 
      cudaMalloc(&d_row_data_, nnz_current * sizeof(index_type)); 
      cudaMalloc(&d_val_data_, nnz_current * sizeof(real_type)); 
      return 0;   
    }
    return -1;
  }

  index_type  MatrixCSC::copyCsc(std::string memspaceOut)
  {

    index_type nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}

    if (memspaceOut == "cpu") {
      //check if we need to copy or not
      if ((d_data_updated_ == true) && (h_data_updated_ == false)) {
        if (h_col_data_ == nullptr) {
          h_col_data_ = new index_type[n_ + 1];      
        }
        if (h_row_data_ == nullptr) {
          h_row_data_ = new index_type[nnz_current];      
        }
        if (h_val_data_ == nullptr) {
          h_val_data_ = new real_type[nnz_current];      
        }
        cudaMemcpy(h_col_data_, d_col_data_, (n_ + 1) * sizeof(index_type), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_row_data_, d_row_data_, nnz_current * sizeof(index_type), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_val_data_, d_val_data_, nnz_current * sizeof(real_type), cudaMemcpyDeviceToHost);
        h_data_updated_ = true;
      }
      return 0;   
    }

    if (memspaceOut == "cuda") {
      if ((d_data_updated_ == false) && (h_data_updated_ == true)) {
        if (d_col_data_ == nullptr) {
          cudaMalloc(&d_col_data_, (n_ + 1) * sizeof(index_type)); 
        }
        if (d_row_data_ == nullptr) {
          cudaMalloc(&d_row_data_, nnz_current * sizeof(index_type)); 
        }
        if (d_val_data_ == nullptr) {
          cudaMalloc(&d_val_data_, nnz_current * sizeof(real_type)); 
        }
        cudaMemcpy(d_col_data_, h_col_data_, (n_ + 1) * sizeof(index_type), cudaMemcpyHostToDevice);
        cudaMemcpy(d_row_data_, h_row_data_, nnz_current * sizeof(index_type), cudaMemcpyHostToDevice);
        cudaMemcpy(d_val_data_, h_val_data_, nnz_current * sizeof(real_type), cudaMemcpyHostToDevice);
        d_data_updated_ = true;
      }
      return 0; 
    }
    return -1;
  }
}
