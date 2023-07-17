#include <cstring>  // <-- includes memcpy
#include <iostream>
#include <iomanip> 

#include <cuda_runtime.h>

#include "MatrixCOO.hpp"


namespace ReSolve 
{
  MatrixCOO::MatrixCOO()
  {
  }

  MatrixCOO::MatrixCOO(Int n, Int m, Int nnz) : Matrix(n, m, nnz)
  {
  }
  
  MatrixCOO::MatrixCOO(Int n, 
                       Int m, 
                       Int nnz,
                       bool symmetric,
                       bool expanded) : Matrix(n, m, nnz, symmetric, expanded)
  {
  }
  
  MatrixCOO::~MatrixCOO()
  {
  }

  Int* MatrixCOO::getRowData(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCoo("cpu");
      return this->h_row_data_;
    } else {
      if (memspace == "cuda") {
        copyCoo("cuda");
        return this->d_row_data_;
      } else {
        return nullptr;
      }
    }
  }

  Int* MatrixCOO::getColData(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCoo("cpu");
      return this->h_col_data_;
    } else {
      if (memspace == "cuda") {
        copyCoo("cuda");
        return this->d_col_data_;
      } else {
        return nullptr;
      }
    }
  }

  Real* MatrixCOO::getValues(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCoo("cpu");
      return this->h_val_data_;
    } else {
      if (memspace == "cuda") {
        copyCoo("cuda");
        return this->d_val_data_;
      } else {
        return nullptr;
      }
    }
  }

  Int MatrixCOO::updateData(Int* row_data, Int* col_data, Real* val_data, std::string memspaceIn, std::string memspaceOut)
  {

    //four cases (for now)
    Int nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    setNotUpdated();
    int control=-1;
    if ((memspaceIn == "cpu") && (memspaceOut == "cpu")){ control = 0;}
    if ((memspaceIn == "cpu") && (memspaceOut == "cuda")){ control = 1;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cpu")){ control = 2;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cuda")){ control = 3;}

    if (memspaceOut == "cpu") {
      //check if cpu data allocated	
      if (h_row_data_ == nullptr) {
        this->h_row_data_ = new Int[nnz_current];
      }
      if (h_col_data_ == nullptr) {
        this->h_col_data_ = new Int[nnz_current];
      }
      if (h_val_data_ == nullptr) {
        this->h_val_data_ = new Real[nnz_current];
      }
    }

    if (memspaceOut == "cuda") {
      //check if cuda data allocated
      if (d_row_data_ == nullptr) {
        cudaMalloc(&d_row_data_, nnz_current * sizeof(Int)); 
      }
      if (d_col_data_ == nullptr) {
        cudaMalloc(&d_col_data_, nnz_current * sizeof(Int)); 
      }
      if (d_val_data_ == nullptr) {
        cudaMalloc(&d_val_data_, nnz_current * sizeof(Real)); 
      }
    }

    switch(control)  {
      case 0: //cpu->cpu
        std::memcpy(h_row_data_, row_data, (nnz_current) * sizeof(Int));
        std::memcpy(h_col_data_, col_data, (nnz_current) * sizeof(Int));
        std::memcpy(h_val_data_, val_data, (nnz_current) * sizeof(Real));
        h_data_updated_ = true;
        break;
      case 2://cuda->cpu
        cudaMemcpy(h_row_data_, row_data, (nnz_current) * sizeof(Int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_col_data_, col_data, (nnz_current) * sizeof(Int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_val_data_, val_data, (nnz_current) * sizeof(Real), cudaMemcpyDeviceToHost);
        h_data_updated_ = true;
        break;
      case 1://cpu->cuda
        cudaMemcpy(d_row_data_, row_data, (nnz_current) * sizeof(Int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_data_, col_data, (nnz_current) * sizeof(Int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_val_data_, val_data, (nnz_current) * sizeof(Real), cudaMemcpyHostToDevice);
        d_data_updated_ = true;
        break;
      case 3://cuda->cuda
        cudaMemcpy(d_row_data_, row_data, (nnz_current) * sizeof(Int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_col_data_, col_data, (nnz_current) * sizeof(Int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_val_data_, val_data, (nnz_current) * sizeof(Real), cudaMemcpyDeviceToDevice);
        d_data_updated_ = true;
        break;
      default:
        return -1;
    }
    return 0;
  } 

  Int MatrixCOO::updateData(Int* row_data, Int* col_data, Real* val_data, Int new_nnz, std::string memspaceIn, std::string memspaceOut)
  {
    this->destroyMatrixData(memspaceOut);
    int i = this->updateData(row_data, col_data, val_data, memspaceIn, memspaceOut);
    return i;
  } 

  Int MatrixCOO::allocateMatrixData(std::string memspace)
  {
    Int nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    destroyMatrixData(memspace);//just in case

    if (memspace == "cpu") {
      this->h_row_data_ = new Int[nnz_current];
      this->h_col_data_ = new Int[nnz_current];
      this->h_val_data_ = new Real[nnz_current];
      return 0;
    }

    if (memspace == "cuda") {
      cudaMalloc(&d_row_data_, nnz_current * sizeof(Int)); 
      cudaMalloc(&d_col_data_, nnz_current * sizeof(Int)); 
      cudaMalloc(&d_val_data_, nnz_current * sizeof(Real)); 
      return 0;
    }
    return -1;
  }

  Int MatrixCOO::copyCoo(std::string memspaceOut)
  {

    Int nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}

    if (memspaceOut == "cpu") {
      //check if we need to copy or not
      if ((d_data_updated_ == true) && (h_data_updated_ == false)) {
        if (h_row_data_ == nullptr) {
          h_row_data_ = new Int[nnz_current];      
        }
        if (h_col_data_ == nullptr) {
          h_col_data_ = new Int[nnz_current];      
        }
        if (h_val_data_ == nullptr) {
          h_val_data_ = new Real[nnz_current];      
        }
        cudaMemcpy(h_row_data_, d_row_data_, nnz_current * sizeof(Int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_col_data_, d_col_data_, nnz_current * sizeof(Int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_val_data_, d_val_data_, nnz_current * sizeof(Real), cudaMemcpyDeviceToHost);
        h_data_updated_ = true;
      }
      return 0;
    }

    if (memspaceOut == "cuda") {
      if ((d_data_updated_ == false) && (h_data_updated_ == true)) {
        if (d_row_data_ == nullptr) {
          cudaMalloc(&d_row_data_, nnz_current *sizeof(Int)); 
        }
        if (d_col_data_ == nullptr) {
          cudaMalloc(&d_col_data_, nnz_current * sizeof(Int)); 
        }
        if (d_val_data_ == nullptr) {
          cudaMalloc(&d_val_data_, nnz_current * sizeof(Real)); 
        }
        cudaMemcpy(d_row_data_, h_row_data_, nnz_current * sizeof(Int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_data_, h_col_data_, nnz_current * sizeof(Int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_val_data_, h_val_data_, nnz_current * sizeof(Real), cudaMemcpyHostToDevice);
        d_data_updated_ = true;
      }
      return 0;
    }
    return -1;
  }

  void MatrixCOO::print()
  {
    std::cout << "  Row:        Column:           Value:\n";
    for(int i = 0; i < nnz_; ++i) {
      std::cout << std::setw(12)  << h_row_data_[i] << " "
                << std::setw(12)  << h_col_data_[i] << " "
                << std::setw(20) << std::setprecision(16) << h_val_data_[i] << "\n";
    }
  }

} // namespace ReSolve
