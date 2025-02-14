#include <cstring>  // <-- includes memcpy
#include <iostream>
#include <iomanip>
#include <cassert>

#include <resolve/utilities/logger/Logger.hpp>
#include "Coo.hpp"


namespace ReSolve 
{
  using out = io::Logger;

  matrix::Coo::Coo()
  {
    sparse_format_ = TRIPLET;
  }

  matrix::Coo::Coo(index_type n, index_type m, index_type nnz) : Sparse(n, m, nnz)
  {
    sparse_format_ = TRIPLET;
  }
  
  matrix::Coo::Coo(index_type n, 
                   index_type m, 
                   index_type nnz,
                   bool symmetric,
                   bool expanded) : Sparse(n, m, nnz, symmetric, expanded)
  {
    sparse_format_ = TRIPLET;
  }
  
  /**
   * @brief Hijacking constructor
   */
  matrix::Coo::Coo(index_type n,
                   index_type m,
                   index_type nnz,
                   bool symmetric,
                   bool expanded,
                   index_type** rows,
                   index_type** cols,
                   real_type** vals,
                   memory::MemorySpace memspaceSrc,
                   memory::MemorySpace memspaceDst)
    : Sparse(n, m, nnz, symmetric, expanded)
  {
    sparse_format_ = TRIPLET;

    int control = -1;
    if ((memspaceSrc == memory::HOST)   && (memspaceDst == memory::HOST))  { control = 0;}
    if ((memspaceSrc == memory::HOST)   && (memspaceDst == memory::DEVICE)){ control = 1;}
    if ((memspaceSrc == memory::DEVICE) && (memspaceDst == memory::HOST))  { control = 2;}
    if ((memspaceSrc == memory::DEVICE) && (memspaceDst == memory::DEVICE)){ control = 3;}

    switch (control)
    {
      case 0: // cpu->cpu
        // Set host data
        h_row_data_ = *rows;
        h_col_data_ = *cols;
        h_val_data_ = *vals;
        h_data_updated_ = true;
        owns_cpu_values_ = true;
        owns_cpu_sparsity_pattern_ = true;
        // Make sure there is no device data.
        if (d_row_data_ || d_col_data_ || d_val_data_) {
          out::error() << "Device data unexpectedly allocated. "
                       << "Possible bug in matrix::Sparse class.\n";
        }
        // Hijack data from the source
        *rows = nullptr;
        *cols = nullptr;
        *vals = nullptr;
        break;
      case 2: // gpu->cpu
        // Set device data and copy it to host
        d_row_data_ = *rows;
        d_col_data_ = *cols;
        d_val_data_ = *vals;
        d_data_updated_ = true;
        owns_gpu_values_ = true;
        owns_gpu_sparsity_pattern_ = true;
        syncData(memspaceDst);
        // Hijack data from the source
        *rows = nullptr;
        *cols = nullptr;
        *vals = nullptr;
        break;
      case 1: // cpu->gpu
        // Set host data and copy it to device
        h_row_data_ = *rows;
        h_col_data_ = *cols;
        h_val_data_ = *vals;
        h_data_updated_ = true;
        owns_cpu_values_ = true;
        owns_cpu_sparsity_pattern_ = true;
        syncData(memspaceDst);
        // Hijack data from the source
        *rows = nullptr;
        *cols = nullptr;
        *vals = nullptr;
        break;
      case 3: // gpu->gpu
        // Set device data
        d_row_data_ = *rows;
        d_col_data_ = *cols;
        d_val_data_ = *vals;
        d_data_updated_ = true;
        owns_gpu_values_ = true;
        owns_gpu_sparsity_pattern_ = true;
        // Make sure there is no device data.
        if (h_row_data_ || h_col_data_ || h_val_data_) {
          out::error() << "Host data unexpectedly allocated. "
                       << "Possible bug in matrix::Sparse class.\n";
        }
        // Hijack data from the source
        *rows = nullptr;
        *cols = nullptr;
        *vals = nullptr;
        break;
      default:
        out::error() << "Coo constructor failed! "
                     << "Possible bug in memory spaces setting.\n";
        break;
    }
  }

  matrix::Coo::~Coo()
  {
  }

  index_type* matrix::Coo::getRowData(memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;

    switch (memspace) {
      case HOST:
        return this->h_row_data_;
      case DEVICE:
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
        return this->h_col_data_;
      case DEVICE:
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
        return this->h_val_data_;
      case DEVICE:
        return this->d_val_data_;
      default:
        return nullptr;
    }
  }

  int matrix::Coo::copyDataFrom(const index_type* row_data,
                                const index_type* col_data,
                                const real_type* val_data,
                                memory::MemorySpace memspaceIn,
                                memory::MemorySpace memspaceOut)
  {

    //four cases (for now)
    index_type nnz_current = nnz_;
    setNotUpdated();
    int control=-1;
    if ((memspaceIn == memory::HOST) && (memspaceOut == memory::HOST)){ control = 0;}
    if ((memspaceIn == memory::HOST) && ((memspaceOut == memory::DEVICE))){ control = 1;}
    if (((memspaceIn == memory::DEVICE)) && (memspaceOut == memory::HOST)){ control = 2;}
    if (((memspaceIn == memory::DEVICE)) && ((memspaceOut == memory::DEVICE))){ control = 3;}

    if (memspaceOut == memory::HOST) {
      //check if cpu data allocated	
      if ((h_row_data_ == nullptr) != (h_col_data_ == nullptr)) {
        if(h_row_data_ == nullptr) {
          out::error() << "In Coo::copyDataFrom host row data is null, but col data is set!\n";
        }
        else {
          out::error() << "In Coo::copyDataFrom host col data is null, but row data is set!\n";
        }
      }
      if ((h_row_data_ == nullptr) && (h_col_data_ == nullptr)) {
        this->h_row_data_ = new index_type[nnz_current];
        this->h_col_data_ = new index_type[nnz_current];
        owns_cpu_sparsity_pattern_ = true;
      }
      if (h_val_data_ == nullptr) {
        this->h_val_data_ = new real_type[nnz_current];
        owns_cpu_values_ = true;
      }
    }

    if (memspaceOut == memory::DEVICE) {
      //check if cuda data allocated
      if ((d_row_data_ == nullptr) != (d_col_data_ == nullptr)) {
        if(d_row_data_ == nullptr) {
          out::error() << "In Coo::copyDataFrom device row data is null, but col data is set!\n";
        }
        else {
          out::error() << "In Coo::copyDataFrom device col data is null, but row data is set!\n";
        }
      }
      if ((d_row_data_ == nullptr) && (d_col_data_ == nullptr)) {
        mem_.allocateArrayOnDevice(&d_row_data_, nnz_current);
        mem_.allocateArrayOnDevice(&d_col_data_, nnz_current);
        owns_gpu_sparsity_pattern_ = true;
      }
      if (d_val_data_ == nullptr) {
        mem_.allocateArrayOnDevice(&d_val_data_, nnz_current);
        owns_gpu_values_ = true;
      }
    }

    switch(control)  {
      case 0: //cpu->cpu
        mem_.copyArrayHostToHost(h_row_data_, row_data, nnz_current);
        mem_.copyArrayHostToHost(h_col_data_, col_data, nnz_current);
        mem_.copyArrayHostToHost(h_val_data_, val_data, nnz_current);
        h_data_updated_ = true;
        break;
      case 2://gpu->cpu
        mem_.copyArrayDeviceToHost(h_row_data_, row_data, nnz_current);
        mem_.copyArrayDeviceToHost(h_col_data_, col_data, nnz_current);
        mem_.copyArrayDeviceToHost(h_val_data_, val_data, nnz_current);
        h_data_updated_ = true;
        break;
      case 1://cpu->gpu
        mem_.copyArrayHostToDevice(d_row_data_, row_data, nnz_current);
        mem_.copyArrayHostToDevice(d_col_data_, col_data, nnz_current);
        mem_.copyArrayHostToDevice(d_val_data_, val_data, nnz_current);
        d_data_updated_ = true;
        break;
      case 3://gpu->gpu
        mem_.copyArrayDeviceToDevice(d_row_data_, row_data, nnz_current);
        mem_.copyArrayDeviceToDevice(d_col_data_, col_data, nnz_current);
        mem_.copyArrayDeviceToDevice(d_val_data_, val_data, nnz_current);
        d_data_updated_ = true;
        break;
      default:
        return -1;
    }
    return 0;
  } 

  int matrix::Coo::copyDataFrom(const index_type* row_data,
                                const index_type* col_data,
                                const real_type* val_data,
                                index_type new_nnz,
                                memory::MemorySpace memspaceIn,
                                memory::MemorySpace memspaceOut)
  {
    destroyMatrixData(memspaceOut);
    nnz_ = new_nnz;
    return copyDataFrom(row_data, col_data, val_data, memspaceIn, memspaceOut);
  } 

  int matrix::Coo::allocateMatrixData(memory::MemorySpace memspace)
  {
    index_type nnz_current = nnz_;
    destroyMatrixData(memspace);//just in case

    if (memspace == memory::HOST) {
      this->h_row_data_ = new index_type[nnz_current];
      std::fill(h_row_data_, h_row_data_ + nnz_current, 0);  
      this->h_col_data_ = new index_type[nnz_current];
      std::fill(h_col_data_, h_col_data_ + nnz_current, 0);  
      this->h_val_data_ = new real_type[nnz_current];
      std::fill(h_val_data_, h_val_data_ + nnz_current, 0.0);  
      owns_cpu_sparsity_pattern_ = true;
      owns_cpu_values_ = true;
      return 0;
    }

    if (memspace == memory::DEVICE) {
      mem_.allocateArrayOnDevice(&d_row_data_, nnz_current); 
      mem_.allocateArrayOnDevice(&d_col_data_, nnz_current); 
      mem_.allocateArrayOnDevice(&d_val_data_, nnz_current); 
      owns_gpu_sparsity_pattern_ = true;
      owns_gpu_values_ = true;
      return 0;
    }
    return -1;
  }

  /**
   * @brief Sync data in memspace with the updated memory space.
   * 
   * @param memspace - memory space to be synced up (HOST or DEVICE)
   * @return int - 0 if successful, error code otherwise
   * 
   * @todo Handle case when neither memory space is updated. Currently,
   * this function does nothing in that situation, quitely ignoring
   * the sync call.
   */
  int matrix::Coo::syncData(memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;

    switch (memspace) {
      case HOST:
        if (h_data_updated_) {
          out::misc() << "WARNING: In Coo::syncData trying to sync host, but host already up to date!"
          << "This line is ignored. (Perhaps you meant to sync device)\n";
          return 0;
        }
        if (!d_data_updated_) {
          out::error() << "In Coo::syncData trying to sync host with device, but device is out of date!" <<
          "If you have changed the data on purpose, update the device with: variableName->setUpdated(memory::DEVICE)."
          << "If you did not mean to change the data on the device, check your code. \n";
          assert(d_data_updated_);
        }
        if ((h_row_data_ == nullptr) != (h_col_data_ == nullptr)) {
          if(h_row_data_ == nullptr) {
            out::error() << "In Coo::syncData host row data is null, but col data is set!\n";
          }
          else {
            out::error() << "In Coo::syncData host col data is null, but row data is set!\n";
          }
        }
        if ((h_row_data_ == nullptr) && (h_col_data_ == nullptr)) {
          h_row_data_ = new index_type[nnz_];      
          h_col_data_ = new index_type[nnz_];      
          owns_cpu_sparsity_pattern_ = true;
        }
        if (h_val_data_ == nullptr) {
          h_val_data_ = new real_type[nnz_];      
          owns_cpu_values_ = true;
        }
        mem_.copyArrayDeviceToHost(h_row_data_, d_row_data_, nnz_);
        mem_.copyArrayDeviceToHost(h_col_data_, d_col_data_, nnz_);
        mem_.copyArrayDeviceToHost(h_val_data_, d_val_data_, nnz_);
        h_data_updated_ = true;
        return 0;
      case DEVICE:
        if (d_data_updated_) {
          out::misc() << "WARNING: In Coo::syncData trying to sync device, but device already up to date!"
          << "This line is ignored. (Perhaps you meant to sync host)\n";
          return 0;
        }
        if (!h_data_updated_) {
          out::error() << "In Coo::syncData trying to sync device with host, but host is out of date!" <<
          "If you have changed the data on purpose, update the host with: variableName->setUpdated(memory::HOST)."
          << "If you did not mean to change the data on the host, check your code. \n";
          assert(h_data_updated_);
        }
        if ((d_row_data_ == nullptr) != (d_col_data_ == nullptr)) {
          if(d_row_data_ == nullptr) {
            out::error() << "In Coo::syncData device row data is null, but col data is set!\n";
          }
          else {
            out::error() << "In Coo::syncData device col data is null, but row data is set!\n";
          }
          
        }
        if ((d_row_data_ == nullptr) && (d_col_data_ == nullptr)) {
          mem_.allocateArrayOnDevice(&d_row_data_, nnz_);
          mem_.allocateArrayOnDevice(&d_col_data_, nnz_);
          owns_gpu_sparsity_pattern_ = true;
        }
        if (d_val_data_ == nullptr) {
          mem_.allocateArrayOnDevice(&d_val_data_, nnz_);
          owns_gpu_values_ = true;
        }
        mem_.copyArrayHostToDevice(d_row_data_, h_row_data_, nnz_);
        mem_.copyArrayHostToDevice(d_col_data_, h_col_data_, nnz_);
        mem_.copyArrayHostToDevice(d_val_data_, h_val_data_, nnz_);
        d_data_updated_ = true;
        return 0;
      default:
        return 1;
    } // switch
  }

  /**
   * @brief Prints matrix data.
   * 
   * @param out - Output stream where the matrix data is printed
   */
  void matrix::Coo::print(std::ostream& out, index_type indexing_base)
  {
    out << std::scientific << std::setprecision(std::numeric_limits<real_type>::digits10);
    for(int i = 0; i < nnz_; ++i) {
      out << h_row_data_[i] + indexing_base << " "
          << h_col_data_[i] + indexing_base << " "
          << h_val_data_[i] << "\n";
    }
  }
} // namespace ReSolve
