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

  index_type* matrix::Coo::getColData(memory::MemorySpace memspace)
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

  real_type* matrix::Coo::getValues(memory::MemorySpace memspace)
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

  int matrix::Coo::updateData(index_type* row_data,
                              index_type* col_data,
                              real_type* val_data,
                              memory::MemorySpace memspaceIn,
                              memory::MemorySpace memspaceOut)
  {

    //four cases (for now)
    index_type nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    setNotUpdated();
    int control=-1;
    if ((memspaceIn == memory::HOST) && (memspaceOut == memory::HOST)){ control = 0;}
    if ((memspaceIn == memory::HOST) && ((memspaceOut == memory::DEVICE))){ control = 1;}
    if (((memspaceIn == memory::DEVICE)) && (memspaceOut == memory::HOST)){ control = 2;}
    if (((memspaceIn == memory::DEVICE)) && ((memspaceOut == memory::DEVICE))){ control = 3;}

    if (memspaceOut == memory::HOST) {
      //check if cpu data allocated
      if ((h_row_data_ == nullptr) != (h_col_data_ == nullptr)) {
        out::error() << "In Coo::updateData one of host row or column data is null!\n";
      }
      if ((h_row_data_ == nullptr) && (h_col_data_ == nullptr)) {
        this->h_row_data_ = new index_type[nnz_current];
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
        out::error() << "In Coo::updateData one of device row or column data is null!\n";
      }
      if ((d_row_data_ == nullptr) && (d_col_data_ == nullptr)) {
        mem_.allocateArrayOnDevice(&d_row_data_, nnz_current);
        mem_.allocateArrayOnDevice(&d_col_data_, nnz_current);
        owns_gpu_data_ = true;
      }
      if (d_val_data_ == nullptr) {
        mem_.allocateArrayOnDevice(&d_val_data_, nnz_current);
        owns_gpu_vals_ = true;
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
      case 3://gpu->gpua
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

  int matrix::Coo::updateData(index_type* row_data, index_type* col_data, real_type* val_data, index_type new_nnz, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut)
  {
    this->destroyMatrixData(memspaceOut);
    this->nnz_ = new_nnz;
    int i = this->updateData(row_data, col_data, val_data, memspaceIn, memspaceOut);
    return i;
  }

  int matrix::Coo::allocateMatrixData(memory::MemorySpace memspace)
  {
    index_type nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    destroyMatrixData(memspace);//just in case

    if (memspace == memory::HOST) {
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

    if (memspace == memory::DEVICE) {
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
          if ((h_row_data_ == nullptr) != (h_col_data_ == nullptr)) {
            out::error() << "In Coo::copyData one of host row or column data is null!\n";
          }
          if ((h_row_data_ == nullptr) && (h_col_data_ == nullptr)) {
            h_row_data_ = new index_type[nnz_current];
            h_col_data_ = new index_type[nnz_current];
            owns_cpu_data_ = true;
          }
          if (h_val_data_ == nullptr) {
            h_val_data_ = new real_type[nnz_current];
            owns_cpu_vals_ = true;
          }
          mem_.copyArrayDeviceToHost(h_row_data_, d_row_data_, nnz_current);
          mem_.copyArrayDeviceToHost(h_col_data_, d_col_data_, nnz_current);
          mem_.copyArrayDeviceToHost(h_val_data_, d_val_data_, nnz_current);
          h_data_updated_ = true;
        }
        return 0;
      case DEVICE:
        if ((d_data_updated_ == false) && (h_data_updated_ == true)) {
          if ((d_row_data_ == nullptr) != (d_col_data_ == nullptr)) {
            out::error() << "In Coo::copyData one of device row or column data is null!\n";
          }
          if ((d_row_data_ == nullptr) && (d_col_data_ == nullptr)) {
            mem_.allocateArrayOnDevice(&d_row_data_, nnz_current);
            mem_.allocateArrayOnDevice(&d_col_data_, nnz_current);
            owns_gpu_data_ = true;
          }
          if (d_val_data_ == nullptr) {
            mem_.allocateArrayOnDevice(&d_val_data_, nnz_current);
            owns_gpu_vals_ = true;
          }
          mem_.copyArrayHostToDevice(d_row_data_, h_row_data_, nnz_current);
          mem_.copyArrayHostToDevice(d_col_data_, h_col_data_, nnz_current);
          mem_.copyArrayHostToDevice(d_val_data_, h_val_data_, nnz_current);
          d_data_updated_ = true;
        }
        return 0;
      default:
        return -1;
    } // switch
  }

  /**
   * @brief Prints matrix data.
   *
   * @param out - Output stream where the matrix data is printed
   */
  void matrix::Coo::print(std::ostream& out)
  {
    out << std::scientific << std::setprecision(std::numeric_limits<real_type>::digits10);
    for(int i = 0; i < nnz_; ++i) {
      out << h_row_data_[i] << " "
          << h_col_data_[i] << " "
          << h_val_data_[i] << "\n";
    }
  }

  int matrix::Coo::expand()
  {
    if (is_symmetric_ && !is_expanded_) {
      index_type* rows = getRowData(memory::HOST);
      index_type* columns = getColData(memory::HOST);
      real_type* values = getValues(memory::HOST);

      if (rows == nullptr || columns == nullptr || values == nullptr) {
        return 0;
      }

      // NOTE: this is predicated on the same define as that which disables
      //       assert(3), to avoid record-keeping where it is not necessary
#ifndef NDEBUG
      index_type n_diagonal = 0;
#endif

      // NOTE: so because most of the code here uses new/delete and there's no
      //       realloc(3) equivalent for that memory management scheme, we
      //       have to manually new/memcpy/delete, unfortunately
      index_type* new_rows = new index_type[nnz_expanded_];
      index_type* new_columns = new index_type[nnz_expanded_];
      real_type* new_values = new real_type[nnz_expanded_];

      index_type j = 0;
      for (index_type i = 0; i < nnz_; i++) {
        new_rows[j] = rows[i];
        new_columns[j] = columns[i];
        new_values[j] = values[i];

        j++;

#ifndef NDEBUG
        if (rows[i] == columns[i]) {
          n_diagonal++;
        } else {
#else
        if (rows[i] != columns[i]) {
#endif
          new_rows[j] = columns[i];
          new_columns[j] = rows[i];
          new_values[j] = values[i];

          j++;
        }
      }

      // NOTE: the effectiveness of this is probably questionable given that
      //       it occurs after we've already risked writing out-of-bounds, but
      //       i guess if that worked or we've over-allocated, this will catch
      //       something (in debug builds/release builds with asserts enabled)
      assert(nnz_expanded_ == ((2 * nnz_) - n_diagonal));

      if (destroyMatrixData(memory::HOST) != 0 ||
          setMatrixData(new_rows, new_columns, new_values, memory::HOST) != 0) {
        // TODO: make fallible
        assert(false && "invalid state after coo matrix expansion");
        return -1;
      }

      setNnz(nnz_expanded_);
      setExpanded(true);
      owns_cpu_data_ = owns_cpu_vals_ = true;
    }

    return 0;
  }
} // namespace ReSolve
