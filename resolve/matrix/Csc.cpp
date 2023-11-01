#include <cstring>  // <-- includes memcpy

#include "Csc.hpp"

namespace ReSolve 
{
  matrix::Csc::Csc()
  {
  }

  matrix::Csc::Csc(index_type n, index_type m, index_type nnz) : Sparse(n, m, nnz)
  {
  }
  
  matrix::Csc::Csc(index_type n, 
                       index_type m, 
                       index_type nnz,
                       bool symmetric,
                       bool expanded) : Sparse(n, m, nnz, symmetric, expanded)
  {
  }

  matrix::Csc::~Csc()
  {
  }

  index_type* matrix::Csc::getRowData(memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    copyData(memspace);
    switch (memspace) {
      case HOST:
        return this->h_row_data_;
      case DEVICE:
        return this->d_row_data_;
      default:
        return nullptr;
    }
  }

  index_type* matrix::Csc::getColData(memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    copyData(memspace);
    switch (memspace) {
      case HOST:
        return this->h_col_data_;
      case DEVICE:
        return this->d_col_data_;
      default:
        return nullptr;
    }
  }

  real_type* matrix::Csc::getValues(memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    copyData(memspace);
    switch (memspace) {
      case HOST:
        return this->h_val_data_;
      case DEVICE:
        return this->d_val_data_;
      default:
        return nullptr;
    }
  }

  int matrix::Csc::updateData(index_type* row_data, index_type* col_data, real_type* val_data, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut)
  {
    index_type nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    //four cases (for now)
    int control=-1;
    setNotUpdated();
    if ((memspaceIn == memory::HOST)     && (memspaceOut == memory::HOST))    { control = 0;}
    if ((memspaceIn == memory::HOST)     && ((memspaceOut == memory::DEVICE))){ control = 1;}
    if (((memspaceIn == memory::DEVICE)) && (memspaceOut == memory::HOST))    { control = 2;}
    if (((memspaceIn == memory::DEVICE)) && ((memspaceOut == memory::DEVICE))){ control = 3;}

    if (memspaceOut == memory::HOST) {
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

    if (memspaceOut == memory::DEVICE) {
      //check if cuda data allocated
      if (d_col_data_ == nullptr) {
        mem_.allocateArrayOnDevice(&d_col_data_, n_ + 1); 
      }
      if (d_row_data_ == nullptr) {
        mem_.allocateArrayOnDevice(&d_row_data_, nnz_current);
      }
      if (d_val_data_ == nullptr) {
        mem_.allocateArrayOnDevice(&d_val_data_, nnz_current); 
      }
    }

    switch(control)  {
      case 0: //cpu->cpu
        mem_.copyArrayHostToHost(h_col_data_, col_data,      n_ + 1);
        mem_.copyArrayHostToHost(h_row_data_, row_data, nnz_current);
        mem_.copyArrayHostToHost(h_val_data_, val_data, nnz_current);
        h_data_updated_ = true;
        owns_cpu_data_ = true;
        owns_cpu_vals_ = true;
        break;
      case 2://gpu->cpu
        mem_.copyArrayDeviceToHost(h_col_data_, col_data,      n_ + 1);
        mem_.copyArrayDeviceToHost(h_row_data_, row_data, nnz_current);
        mem_.copyArrayDeviceToHost(h_val_data_, val_data, nnz_current);
        h_data_updated_ = true;
        owns_cpu_data_ = true;
        owns_cpu_vals_ = true;
        break;
      case 1://cpu->gpu
        mem_.copyArrayHostToDevice(d_col_data_, col_data,      n_ + 1);
        mem_.copyArrayHostToDevice(d_row_data_, row_data, nnz_current);
        mem_.copyArrayHostToDevice(d_val_data_, val_data, nnz_current);
        d_data_updated_ = true;
        owns_gpu_data_ = true;
        owns_gpu_vals_ = true;
        break;
      case 3://gpu->gpu
        mem_.copyArrayDeviceToDevice(d_col_data_, col_data,      n_ + 1);
        mem_.copyArrayDeviceToDevice(d_row_data_, row_data, nnz_current);
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

  int matrix::Csc::updateData(index_type* row_data, index_type* col_data, real_type* val_data, index_type new_nnz, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut)
  {
    this->destroyMatrixData(memspaceOut);
    this->nnz_ = new_nnz;
    int i = this->updateData(col_data, row_data, val_data, memspaceIn, memspaceOut);
    return i;
  }

  int matrix::Csc::allocateMatrixData(memory::MemorySpace memspace)
  {
    index_type nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    destroyMatrixData(memspace);//just in case

    if (memspace == memory::HOST) {
      this->h_col_data_ = new index_type[n_ + 1];
      std::fill(h_col_data_, h_col_data_ + n_ + 1, 0);  
      this->h_row_data_ = new index_type[nnz_current];
      std::fill(h_row_data_, h_row_data_ + nnz_current, 0);  
      this->h_val_data_ = new real_type[nnz_current];
      std::fill(h_val_data_, h_val_data_ + nnz_current, 0.0);  
      owns_cpu_data_ = true;
      owns_cpu_vals_ = true;
      return 0;
    }

    if (memspace == memory::DEVICE) {
      mem_.allocateArrayOnDevice(&d_col_data_,      n_ + 1); 
      mem_.allocateArrayOnDevice(&d_row_data_, nnz_current); 
      mem_.allocateArrayOnDevice(&d_val_data_, nnz_current); 
      owns_gpu_data_ = true;
      owns_gpu_vals_ = true;
      return 0;   
    }
    return -1;
  }

  int matrix::Csc::copyData(memory::MemorySpace memspaceOut)
  {
    using namespace ReSolve::memory;

    index_type nnz_current = nnz_;
    if (is_expanded_) {
      nnz_current = nnz_expanded_;
    }

    switch(memspaceOut) {
      case HOST:
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
          mem_.copyArrayDeviceToHost(h_col_data_, d_col_data_,      n_ + 1);
          mem_.copyArrayDeviceToHost(h_row_data_, d_row_data_, nnz_current);
          mem_.copyArrayDeviceToHost(h_val_data_, d_val_data_, nnz_current);
          h_data_updated_ = true;
          owns_cpu_data_ = true;
          owns_cpu_vals_ = true;
        }
        return 0;   
      case DEVICE:
        if ((d_data_updated_ == false) && (h_data_updated_ == true)) {
          if (d_col_data_ == nullptr) {
            mem_.allocateArrayOnDevice(&d_col_data_, n_ + 1); 
          }
          if (d_row_data_ == nullptr) {
            mem_.allocateArrayOnDevice(&d_row_data_, nnz_current); 
          }
          if (d_val_data_ == nullptr) {
            mem_.allocateArrayOnDevice(&d_val_data_, nnz_current); 
          }
          mem_.copyArrayHostToDevice(d_col_data_, h_col_data_,      n_ + 1);
          mem_.copyArrayHostToDevice(d_row_data_, h_row_data_, nnz_current);
          mem_.copyArrayHostToDevice(d_val_data_, h_val_data_, nnz_current);
          d_data_updated_ = true;
          owns_gpu_data_ = true;
          owns_gpu_vals_ = true;
        }
        return 0;
      default:
        return -1;
    } // switch
  }
}
