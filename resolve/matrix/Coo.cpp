#include <cstring>  // <-- includes memcpy
#include <iostream>
#include <iomanip> 

#include "Coo.hpp"


namespace ReSolve 
{
  matrix::Coo::Coo()
  {
  }

  matrix::Coo::Coo(index_type n, index_type m, index_type nnz) : Sparse(n, m, nnz)
  {
  }
  
  matrix::Coo::Coo(index_type n, 
                   index_type m, 
                   index_type nnz,
                   bool symmetric,
                   bool expanded) : Sparse(n, m, nnz, symmetric, expanded)
  {
  }
  
  matrix::Coo::~Coo()
  {
  }

  index_type* matrix::Coo::getRowData(memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        copyData(memspace);
        return this->h_row_data_;
      case DEVICE:
        copyData(memspace);
        return this->d_row_data_;
      default:
        return nullptr;
    }
  }

  index_type* matrix::Coo::getColData(memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        copyData(memspace);
        return this->h_col_data_;
      case DEVICE:
        copyData(memspace);
        return this->d_col_data_;
      default:
        return nullptr;
    }
  }

  real_type* matrix::Coo::getValues(memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        copyData(memspace);
        return this->h_val_data_;
      case DEVICE:
        copyData(memspace);
        return this->d_val_data_;
      default:
        return nullptr;
    }
  }

  index_type matrix::Coo::updateData(index_type* row_data, index_type* col_data, real_type* val_data, std::string memspaceIn, std::string memspaceOut)
  {

    //four cases (for now)
    index_type nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    setNotUpdated();
    int control=-1;
    if ((memspaceIn == "cpu") && (memspaceOut == "cpu")){ control = 0;}
    if ((memspaceIn == "cpu") && ((memspaceOut == "cuda") || (memspaceOut == "hip"))){ control = 1;}
    if (((memspaceIn == "cuda") || (memspaceIn == "hip")) && (memspaceOut == "cpu")){ control = 2;}
    if (((memspaceIn == "cuda") || (memspaceIn == "hip")) && ((memspaceOut == "cuda") || (memspaceOut == "hip"))){ control = 3;}

    if (memspaceOut == "cpu") {
      //check if cpu data allocated	
      if (h_row_data_ == nullptr) {
        this->h_row_data_ = new index_type[nnz_current];
      }
      if (h_col_data_ == nullptr) {
        this->h_col_data_ = new index_type[nnz_current];
      }
      if (h_val_data_ == nullptr) {
        this->h_val_data_ = new real_type[nnz_current];
      }
    }

    if ((memspaceOut == "cuda") || (memspaceOut == "hip")) {
      //check if cuda data allocated
      if (d_row_data_ == nullptr) {
        mem_.allocateArrayOnDevice(&d_row_data_, nnz_current);
      }
      if (d_col_data_ == nullptr) {
        mem_.allocateArrayOnDevice(&d_col_data_, nnz_current);
      }
      if (d_val_data_ == nullptr) {
        mem_.allocateArrayOnDevice(&d_val_data_, nnz_current);
      }
    }

    switch(control)  {
      case 0: //cpu->cpu
        mem_.copyArrayHostToHost(h_row_data_, row_data, nnz_current);
        mem_.copyArrayHostToHost(h_col_data_, col_data, nnz_current);
        mem_.copyArrayHostToHost(h_val_data_, val_data, nnz_current);
        h_data_updated_ = true;
        owns_cpu_data_ = true;
        owns_cpu_vals_ = true;
        break;
      case 2://gpu->cpu
        mem_.copyArrayDeviceToHost(h_row_data_, row_data, nnz_current);
        mem_.copyArrayDeviceToHost(h_col_data_, col_data, nnz_current);
        mem_.copyArrayDeviceToHost(h_val_data_, val_data, nnz_current);
        h_data_updated_ = true;
        owns_cpu_data_ = true;
        owns_cpu_vals_ = true;
        break;
      case 1://cpu->gpu
        mem_.copyArrayHostToDevice(d_row_data_, row_data, nnz_current);
        mem_.copyArrayHostToDevice(d_col_data_, col_data, nnz_current);
        mem_.copyArrayHostToDevice(d_val_data_, val_data, nnz_current);
        d_data_updated_ = true;
        owns_gpu_data_ = true;
        owns_gpu_vals_ = true;
        break;
      case 3://gpu->gpua
        mem_.copyArrayDeviceToDevice(d_row_data_, row_data, nnz_current);
        mem_.copyArrayDeviceToDevice(d_col_data_, col_data, nnz_current);
        mem_.copyArrayDeviceToDevice(d_val_data_, val_data, nnz_current);
        d_data_updated_ = true;
        owns_gpu_data_ = true;
        owns_gpu_vals_ = true;
        break;
      default:
        return -1;
    }
    return 0;
  } 

  index_type matrix::Coo::updateData(index_type* row_data, index_type* col_data, real_type* val_data, index_type new_nnz, std::string memspaceIn, std::string memspaceOut)
  {
    this->destroyMatrixData(memspaceOut);
    this->nnz_ = new_nnz;
    int i = this->updateData(row_data, col_data, val_data, memspaceIn, memspaceOut);
    return i;
  } 

  index_type matrix::Coo::allocateMatrixData(std::string memspace)
  {
    index_type nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    destroyMatrixData(memspace);//just in case

    if (memspace == "cpu") {
      this->h_row_data_ = new index_type[nnz_current];
      std::fill(h_row_data_, h_row_data_ + nnz_current, 0);  
      this->h_col_data_ = new index_type[nnz_current];
      std::fill(h_col_data_, h_col_data_ + nnz_current, 0);  
      this->h_val_data_ = new real_type[nnz_current];
      std::fill(h_val_data_, h_val_data_ + nnz_current, 0.0);  
      owns_cpu_data_ = true;
      owns_cpu_vals_ = true;
      return 0;
    }

    if ((memspace == "cuda") || (memspace == "hip")) {
      mem_.allocateArrayOnDevice(&d_row_data_, nnz_current); 
      mem_.allocateArrayOnDevice(&d_col_data_, nnz_current); 
      mem_.allocateArrayOnDevice(&d_val_data_, nnz_current); 
      owns_gpu_data_ = true;
      owns_gpu_vals_ = true;
      return 0;
    }
    return -1;
  }

  int matrix::Coo::copyData(memory::MemorySpace memspaceOut)
  {
    using namespace ReSolve::memory;

    index_type nnz_current = nnz_;
    if (is_expanded_) {
      nnz_current = nnz_expanded_;
    }

    switch (memspaceOut) {
      case HOST:
        if ((d_data_updated_ == true) && (h_data_updated_ == false)) {
          if (h_row_data_ == nullptr) {
            h_row_data_ = new index_type[nnz_current];      
          }
          if (h_col_data_ == nullptr) {
            h_col_data_ = new index_type[nnz_current];      
          }
          if (h_val_data_ == nullptr) {
            h_val_data_ = new real_type[nnz_current];      
          }
          mem_.copyArrayDeviceToHost(h_row_data_, d_row_data_, nnz_current);
          mem_.copyArrayDeviceToHost(h_col_data_, d_col_data_, nnz_current);
          mem_.copyArrayDeviceToHost(h_val_data_, d_val_data_, nnz_current);
          h_data_updated_ = true;
          owns_cpu_data_ = true;
          owns_cpu_vals_ = true;
        }
        return 0;
      case DEVICE:
        if ((d_data_updated_ == false) && (h_data_updated_ == true)) {
          if (d_row_data_ == nullptr) {
            mem_.allocateArrayOnDevice(&d_row_data_, nnz_current);
          }
          if (d_col_data_ == nullptr) {
            mem_.allocateArrayOnDevice(&d_col_data_, nnz_current);
          }
          if (d_val_data_ == nullptr) {
            mem_.allocateArrayOnDevice(&d_val_data_, nnz_current);
          }
          mem_.copyArrayHostToDevice(d_row_data_, h_row_data_, nnz_current);
          mem_.copyArrayHostToDevice(d_col_data_, h_col_data_, nnz_current);
          mem_.copyArrayHostToDevice(d_val_data_, h_val_data_, nnz_current);
          d_data_updated_ = true;
          owns_gpu_data_ = true;
          owns_gpu_vals_ = true;
        }
        return 0;
    } // switch
  }

  void matrix::Coo::print()
  {
    std::cout << "  Row:        Column:           Value:\n";
    for(int i = 0; i < nnz_; ++i) {
      std::cout << std::setw(12)  << h_row_data_[i] << " "
                << std::setw(12)  << h_col_data_[i] << " "
                << std::setw(20) << std::setprecision(16) << h_val_data_[i] << "\n";
    }
  }

} // namespace ReSolve
