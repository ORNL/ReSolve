#include <cstring>  // <-- includes memcpy
#include <algorithm>
#include <cassert>
#include <iomanip>

#include "Csr.hpp"
#include "Coo.hpp"
#include <resolve/utilities/logger/Logger.hpp>

namespace ReSolve 
{
  using out = io::Logger;

  matrix::Csr::Csr()
  {
    sparse_format_ = COMPRESSED_SPARSE_ROW;
  }

  matrix::Csr::Csr(index_type n, index_type m, index_type nnz) : Sparse(n, m, nnz)
  {
    sparse_format_ = COMPRESSED_SPARSE_ROW;
  }
  
  matrix::Csr::Csr(index_type n, 
                   index_type m, 
                   index_type nnz,
                   bool symmetric,
                   bool expanded) : Sparse(n, m, nnz, symmetric, expanded)
  {
    sparse_format_ = COMPRESSED_SPARSE_ROW;
  }

  /**
   * @brief Hijacking constructor
   * 
   * @param[in] n 
   * @param[in] m 
   * @param[in] nnz 
   * @param[in] symmetric 
   * @param[in] expanded 
   * @param[in,out] rows 
   * @param[in,out] cols 
   * @param[in,out] vals 
   * @param[in] memspaceSrc 
   * @param[in] memspaceDst 
   */
  matrix::Csr::Csr(index_type n, 
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
    sparse_format_ = COMPRESSED_SPARSE_ROW;

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
        owns_cpu_data_  = true;
        // Set device data to null
        if (d_row_data_ || d_col_data_ || d_val_data_) {
          out::error() << "Device data unexpectedly allocated. "
                       << "Possible bug in matrix::Sparse class.\n";
        }
        d_row_data_ = nullptr;
        d_col_data_ = nullptr;
        d_val_data_ = nullptr;
        d_data_updated_ = false;
        owns_gpu_data_  = false;
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
        owns_gpu_data_  = true;
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
        owns_cpu_data_  = true;
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
        owns_gpu_data_  = true;
        // Set host data to null
        if (h_row_data_ || h_col_data_ || h_val_data_) {
          out::error() << "Host data unexpectedly allocated. "
                       << "Possible bug in matrix::Sparse class.\n";
        }
        h_row_data_ = nullptr;
        h_col_data_ = nullptr;
        h_val_data_ = nullptr;
        h_data_updated_ = false;
        owns_cpu_data_  = false;
        // Hijack data from the source
        *rows = nullptr;
        *cols = nullptr;
        *vals = nullptr;
        break;
      default:
        out::error() << "Csr constructor failed! "
                     << "Possible bug in memory spaces setting.\n";
        break;
    }
  }

  matrix::Csr::~Csr()
  {
  }

  index_type* matrix::Csr::getRowData(memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    // syncData(memspace);
    switch (memspace) {
      case HOST:
        return this->h_row_data_;
      case DEVICE:
        return this->d_row_data_;
      default:
        return nullptr;
    }
  }

  index_type* matrix::Csr::getColData(memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    // syncData(memspace);
    switch (memspace) {
      case HOST:
        return this->h_col_data_;
      case DEVICE:
        return this->d_col_data_;
      default:
        return nullptr;
    }
  }

  real_type* matrix::Csr::getValues(memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    // syncData(memspace);
    switch (memspace) {
      case HOST:
        return this->h_val_data_;
      case DEVICE:
        return this->d_val_data_;
      default:
        return nullptr;
    }
  }

  int matrix::Csr::updateData(index_type* row_data, index_type* col_data, real_type* val_data, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut)
  {
    //four cases (for now)
    index_type nnz_current = nnz_;
    setNotUpdated();
    int control = -1;
    if ((memspaceIn == memory::HOST)     && (memspaceOut == memory::HOST))    { control = 0;}
    if ((memspaceIn == memory::HOST)     && ((memspaceOut == memory::DEVICE))){ control = 1;}
    if (((memspaceIn == memory::DEVICE)) && (memspaceOut == memory::HOST))    { control = 2;}
    if (((memspaceIn == memory::DEVICE)) && ((memspaceOut == memory::DEVICE))){ control = 3;}

    if (memspaceOut == memory::HOST) {
      //check if cpu data allocated
      if ((h_row_data_ == nullptr) != (h_col_data_ == nullptr)) {
        out::error() << "In Csr::updateData one of host row or column data is null!\n";
      }
      if ((h_row_data_ == nullptr) && (h_col_data_ == nullptr)) {
        this->h_row_data_ = new index_type[n_ + 1];
        this->h_col_data_ = new index_type[nnz_current];
        owns_cpu_data_ = true;
      } 
      if (h_val_data_ == nullptr) {
        this->h_val_data_ = new real_type[nnz_current];
        owns_cpu_vals_ = true;
      }
    }

    if (memspaceOut == memory::DEVICE) {
      //check if cuda data allocated
      if ((d_row_data_ == nullptr) != (d_col_data_ == nullptr)) {
        out::error() << "In Csr::updateData one of device row or column data is null!\n";
      }
      if ((d_row_data_ == nullptr) && (d_col_data_ == nullptr)) {
        mem_.allocateArrayOnDevice(&d_row_data_, n_ + 1); 
        mem_.allocateArrayOnDevice(&d_col_data_, nnz_current);
        owns_gpu_vals_ = true;
      }
      if (d_val_data_ == nullptr) {
        mem_.allocateArrayOnDevice(&d_val_data_, nnz_current); 
        owns_gpu_data_ = true;
      }
    }


    //copy	
    switch(control)  {
      case 0: //cpu->cpu
        mem_.copyArrayHostToHost(h_row_data_, row_data,      n_ + 1);
        mem_.copyArrayHostToHost(h_col_data_, col_data, nnz_current);
        mem_.copyArrayHostToHost(h_val_data_, val_data, nnz_current);
        h_data_updated_ = true;
        break;
      case 2://gpu->cpu
        mem_.copyArrayDeviceToHost(h_row_data_, row_data,      n_ + 1);
        mem_.copyArrayDeviceToHost(h_col_data_, col_data, nnz_current);
        mem_.copyArrayDeviceToHost(h_val_data_, val_data, nnz_current);
        h_data_updated_ = true;
        break;
      case 1://cpu->gpu
        mem_.copyArrayHostToDevice(d_row_data_, row_data,      n_ + 1);
        mem_.copyArrayHostToDevice(d_col_data_, col_data, nnz_current);
        mem_.copyArrayHostToDevice(d_val_data_, val_data, nnz_current);
        d_data_updated_ = true;
        break;
      case 3://gpu->gpu
        mem_.copyArrayDeviceToDevice(d_row_data_, row_data,      n_ + 1);
        mem_.copyArrayDeviceToDevice(d_col_data_, col_data, nnz_current);
        mem_.copyArrayDeviceToDevice(d_val_data_, val_data, nnz_current);
        d_data_updated_ = true;
        break;
      default:
        return -1;
    }
    return 0;
  } 

  int matrix::Csr::updateData(index_type* row_data, index_type* col_data, real_type* val_data, index_type new_nnz, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut)
  {
    this->destroyMatrixData(memspaceOut);
    this->nnz_ = new_nnz;
    int i = this->updateData(row_data, col_data, val_data, memspaceIn, memspaceOut);
    return i;
  } 

  int matrix::Csr::allocateMatrixData(memory::MemorySpace memspace)
  {
    index_type nnz_current = nnz_;
    destroyMatrixData(memspace);//just in case

    if (memspace == memory::HOST) {
      this->h_row_data_ = new index_type[n_ + 1];
      std::fill(h_row_data_, h_row_data_ + n_ + 1, 0);  
      this->h_col_data_ = new index_type[nnz_current];
      std::fill(h_col_data_, h_col_data_ + nnz_current, 0);  
      this->h_val_data_ = new real_type[nnz_current];
      std::fill(h_val_data_, h_val_data_ + nnz_current, 0.0);  
      owns_cpu_data_ = true;
      owns_cpu_vals_ = true;
      return 0;   
    }

    if (memspace == memory::DEVICE) {
      mem_.allocateArrayOnDevice(&d_row_data_,      n_ + 1); 
      mem_.allocateArrayOnDevice(&d_col_data_, nnz_current); 
      mem_.allocateArrayOnDevice(&d_val_data_, nnz_current); 
      owns_gpu_data_ = true;
      owns_gpu_vals_ = true;
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
  int matrix::Csr::syncData(memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;

    switch (memspace) {
      case HOST:
        //check if we need to copy or not
        if (h_data_updated_) {
          out::misc() << "In Csr::syncData trying to sync host, but host already up to date!\n";
          return 0;
        }
        if (!d_data_updated_) {
          out::error() << "In Csr::syncData trying to sync host with device, but device is out of date!\n";
          assert(d_data_updated_);
        }
        if ((h_row_data_ == nullptr) != (h_col_data_ == nullptr)) {
          out::error() << "In Csr::syncData one of host row or column data is null!\n";
        }
        if ((h_row_data_ == nullptr) && (h_col_data_ == nullptr)) {
          h_row_data_ = new index_type[n_ + 1];
          h_col_data_ = new index_type[nnz_];      
          owns_cpu_data_ = true;
        }
        if (h_val_data_ == nullptr) {
          h_val_data_ = new real_type[nnz_];      
          owns_cpu_vals_ = true;
        }
        mem_.copyArrayDeviceToHost(h_row_data_, d_row_data_, n_ + 1);
        mem_.copyArrayDeviceToHost(h_col_data_, d_col_data_, nnz_);
        mem_.copyArrayDeviceToHost(h_val_data_, d_val_data_, nnz_);
        h_data_updated_ = true;
        return 0;
      case DEVICE:
        if (d_data_updated_) {
          out::misc() << "In Csr::syncData trying to sync device, but device already up to date!\n";
          return 0;
        }
        if (!h_data_updated_) {
          out::error() << "In Csr::syncData trying to sync device with host, but host is out of date!\n";
          assert(h_data_updated_);
        }
        if ((d_row_data_ == nullptr) != (d_col_data_ == nullptr)) {
          out::error() << "In Csr::syncData one of device row or column data is null!\n";
        }
        if ((d_row_data_ == nullptr) && (d_col_data_ == nullptr)) {
          mem_.allocateArrayOnDevice(&d_row_data_, n_ + 1); 
          mem_.allocateArrayOnDevice(&d_col_data_, nnz_); 
          owns_gpu_data_ = true;
        }
        if (d_val_data_ == nullptr) {
          mem_.allocateArrayOnDevice(&d_val_data_, nnz_); 
          owns_gpu_vals_ = true;
        }
        mem_.copyArrayHostToDevice(d_row_data_, h_row_data_, n_ + 1);
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
  void matrix::Csr::print(std::ostream& out, index_type indexing_base)
  {
    out << std::scientific << std::setprecision(std::numeric_limits<real_type>::digits10);
    for(index_type i = 0; i < n_; ++i) {
      for (index_type j = h_row_data_[i]; j < h_row_data_[i+1]; ++j) {
        out << i              + indexing_base << " " 
            << h_col_data_[j] + indexing_base << " "
            << h_val_data_[j] << "\n";
      }
    }
  }
} // namespace ReSolve 

