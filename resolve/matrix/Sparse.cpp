#include <cstring>  // <-- includes memcpy

#include <resolve/utilities/logger/Logger.hpp>
#include "Sparse.hpp"

namespace ReSolve {

  using out = io::Logger;

  /** 
   * @brief empty constructor that does absolutely nothing        
   */
  matrix::Sparse::Sparse()
  {
  }

  /** 
   * @brief basic constructor. It DOES NOT allocate any memory!
   *
   * @param[in] n   - number of rows
   * @param[in] m   - number of columns
   * @param[in] nnz - number of non-zeros        
   */
  matrix::Sparse::Sparse(index_type n, 
                         index_type m, 
                         index_type nnz):
    n_{n},
    m_{m},
    nnz_{nnz}
  {
    this->is_symmetric_ = false;
    this->is_expanded_ = true; //default is a normal non-symmetric fully expanded matrix

    setNotUpdated();

    //set everything to nullptr
    h_row_data_ = nullptr;
    h_col_data_ = nullptr;
    h_val_data_ = nullptr;

    d_row_data_ = nullptr;
    d_col_data_ = nullptr;
    d_val_data_ = nullptr;
    
    owns_cpu_data_ = false;
    owns_cpu_vals_ = false;
    
    owns_gpu_data_ = false;
    owns_gpu_vals_ = false;
  }

  /** 
   * @brief another basic constructor. It DOES NOT allocate any memory!
   *
   * @param[in] n         - number of rows
   * @param[in] m         - number of columns
   * @param[in] nnz       - number of non-zeros        
   * @param[in] symmetric - true if symmetric, false if non-symmetric       
   * @param[in] expanded  - true if expanded, false if not       
   */
  matrix::Sparse::Sparse(index_type n, 
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
    setNotUpdated();

    //set everything to nullptr
    h_row_data_ = nullptr;
    h_col_data_ = nullptr;
    h_val_data_ = nullptr;

    d_row_data_ = nullptr;
    d_col_data_ = nullptr;
    d_val_data_ = nullptr;

    owns_cpu_data_ = false;
    owns_cpu_vals_ = false;
    
    owns_gpu_data_ = false;
    owns_gpu_vals_ = false;
  }
  
  /**
   * @brief destructor
   * */
  matrix::Sparse::~Sparse()
  {
    this->destroyMatrixData(memory::HOST);
    this->destroyMatrixData(memory::DEVICE);
  }

  /** 
   * @brief set the matrix update flags to false (for both HOST and DEVICE).
   */
  void matrix::Sparse::setNotUpdated()
  {
    h_data_updated_ = false;
    d_data_updated_ = false; 
  }
  
  /**
   * @brief get number of matrix rows
   *
   * @return number of matrix rows.
   */
  index_type matrix::Sparse::getNumRows()
  {
    return this->n_;
  }

  /**
   * @brief get number of matrix columns
   *
   * @return number of matrix columns.
   */
  index_type matrix::Sparse::getNumColumns()
  {
    return this->m_;
  }

  /**
   * @brief get number of non-zeros in the matrix.
   *
   * @return number of non-zeros.
   */
  index_type matrix::Sparse::getNnz()
  {
    return this->nnz_;
  }

  matrix::Sparse::SparseFormat matrix::Sparse::getSparseFormat() const
  {
    return sparse_format_;
  }

  /**
   * @brief check if matrix is symmetric.
   *
   * @return true if symmetric, false otherwise.
   */
  bool matrix::Sparse::symmetric()
  {
    return is_symmetric_;
  }

  /**
   * @brief check if (symmetric) matrix is expanded.
   *
   * @return true if expanded, false otherwise.
   */
  bool matrix::Sparse::expanded()
  {
    return is_expanded_;
  }

  /**
   * @brief Set matrix symmetry property
   *
   * @param[in] symmetric - true to set matrix to symmetric and false to set to non-symmetric 
   */  
  void matrix::Sparse::setSymmetric(bool symmetric)
  {
    this->is_symmetric_ = symmetric;
  }

  /**
   * @brief Set matrix "expanded" property
   *
   * @param[in] expanded - true to set matrix to expanded and false to set to not expanded
   */  
  void matrix::Sparse::setExpanded(bool expanded)
  {
    this->is_expanded_ = expanded;
  }

  /**
   * @brief Set number of non-zeros.
   *
   * @param[in] nnz_new - new number of non-zeros
   */  
  void matrix::Sparse::setNnz(index_type nnz_new)
  {
    this->nnz_ = nnz_new;
  }

  /**
   * @brief Set the data to be updated on HOST or DEVICE. 
   *
   * @param[in] memspace - memory space (HOST or DEVICE) of data that is set to "updated"
   *
   * @return 0 if successful, -1 if not.
   * 
   * @note The method automatically sets the other mirror data to non-updated (but it does not copy).
   */  
  int matrix::Sparse::setUpdated(memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        h_data_updated_ = true;
        d_data_updated_ = false;
        break;
      case DEVICE:
        d_data_updated_ = true;
        h_data_updated_ = false;
        break;
    }
    return 0;
  }

  /**
   * @brief Set the pointers for matrix row, column, value data.
   * 
   * Useful if interfacing with other codes - this function only assigns
   * pointers, but it does not allocate nor copy anything. The data ownership
   * flags would be set to false (default).
   *
   * @param[in] row_data - pointer to row data (array of integers)
   * @param[in] col_data - pointer to column data (array of integers)
   * @param[in] val_data - pointer to value data (array of real numbers)
   * @param[in] memspace - memory space (HOST or DEVICE) of incoming data
   *
   * @return 0 if successful, 1 if not.
   */  
  int matrix::Sparse::setDataPointers(index_type* row_data,
                                      index_type* col_data,
                                      real_type*  val_data,
                                      memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;

    setNotUpdated();

    switch (memspace) {
      case HOST:
        if (owns_cpu_data_ && (h_row_data_ || h_col_data_)) {
          out::error() << "Trying to set matrix host data, but the data already set!\n";
          out::error() << "Ignoring setDataPointers function call ...\n";
          return 1;
        }
        if (owns_cpu_vals_ && h_val_data_) {
          out::error() << "Trying to set matrix host values, but the values already set!\n";
          out::error() << "Ignoring setValuesPointer function call ...\n";
          return 1;
        }
        h_row_data_ = row_data;
        h_col_data_ = col_data;
        h_val_data_ = val_data;	
        h_data_updated_ = true;
        owns_cpu_data_  = false;
        owns_cpu_vals_  = false;
        break;
      case DEVICE:
        if (owns_gpu_data_ && (d_row_data_ || d_col_data_)) {
          out::error() << "Trying to set matrix host data, but the data already set!\n";
          out::error() << "Ignoring setDataPointers function call ...\n";
          return 1;
        }
        if (owns_gpu_vals_ && d_val_data_) {
          out::error() << "Trying to set matrix device values, but the values already set!\n";
          out::error() << "Ignoring setValuesPointer function call ...\n";
          return 1;
        }
        d_row_data_ = row_data;
        d_col_data_ = col_data;
        d_val_data_ = val_data;	
        d_data_updated_ = true;
        owns_gpu_data_  = false;
        owns_gpu_vals_  = false;
        break;
    }
    return 0;
  }
  
  /**
   * @brief destroy matrix data (HOST or DEVICE) if the matrix owns it 
   * (will attempt to destroy all three arrays).
   *
   * @param[in] memspace - memory space (HOST or DEVICE) of incoming data
   *
   * @return 0 if successful, -1 if not.
   *
   */ 
  int matrix::Sparse::destroyMatrixData(memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        if (owns_cpu_data_) {
          delete [] h_row_data_;
          delete [] h_col_data_;
          h_row_data_ = nullptr;
          h_col_data_ = nullptr;
        }
        if (owns_cpu_vals_) {
          delete [] h_val_data_;
          h_val_data_ = nullptr;
        }
        return 0;
      case DEVICE:
        if (owns_gpu_data_) {
          mem_.deleteOnDevice(d_row_data_);
          mem_.deleteOnDevice(d_col_data_);
          d_row_data_ = nullptr;
          d_col_data_ = nullptr;
        }
        if (owns_gpu_vals_) {
          mem_.deleteOnDevice(d_val_data_);
          d_val_data_ = nullptr;
        }
        return 0;
      default:
        return -1;
    }
  }

  /**
   * @brief updata matrix values using the _new_values_ provided either as HOST or as DEVICE array.
   * 
   * This function will copy the data (not just assign a pointer) and allocate if needed.
   * It also sets ownership and update flags.
   *
   * @param[in] new_vals    - pointer to new values data (array of real numbers)
   * @param[in] memspaceIn  - memory space (HOST or DEVICE) of _new_vals_
   * @param[in] memspaceOut - memory space (HOST or DEVICE) of matrix values to be updated.
   *
   * @return 0 if successful, -1 if not.
   */  
  int matrix::Sparse::updateValues(const real_type* new_vals,
                                   memory::MemorySpace memspaceIn,
                                   memory::MemorySpace memspaceOut)
  {
 
    index_type nnz_current = nnz_;
    //four cases (for now)
    setNotUpdated();
    int control=-1;
    if ((memspaceIn == memory::HOST)   && (memspaceOut == memory::HOST))  { control = 0;}
    if ((memspaceIn == memory::HOST)   && (memspaceOut == memory::DEVICE)){ control = 1;}
    if ((memspaceIn == memory::DEVICE) && (memspaceOut == memory::HOST))  { control = 2;}
    if ((memspaceIn == memory::DEVICE) && (memspaceOut == memory::DEVICE)){ control = 3;}
   
    if (memspaceOut == memory::HOST) {
      //check if cpu data allocated
      if (h_val_data_ == nullptr) {
        this->h_val_data_ = new real_type[nnz_current];
        owns_cpu_vals_ = true;
      }
    }

    if (memspaceOut == memory::DEVICE) {
      //check if cuda data allocated
      if (d_val_data_ == nullptr) {
        mem_.allocateArrayOnDevice(&d_val_data_, nnz_current); 
        owns_gpu_vals_ = true;
      }
    }

    switch(control)  {
      case 0: //cpu->cpu
        mem_.copyArrayHostToHost(h_val_data_, new_vals, nnz_current);
        h_data_updated_ = true;
        break;
      case 2://cuda->cpu
        mem_.copyArrayDeviceToHost(h_val_data_, new_vals, nnz_current);
        h_data_updated_ = true;
        break;
      case 1://cpu->cuda
        mem_.copyArrayHostToDevice(d_val_data_, new_vals, nnz_current);
        d_data_updated_ = true;
        break;
      case 3://cuda->cuda
        mem_.copyArrayDeviceToDevice(d_val_data_, new_vals, nnz_current);
        d_data_updated_ = true;
        break;
      default:
        return -1;
    }
    return 0;
  }

  /**
   * @brief updata matrix values using the _new_values_ provided either as
   * HOST or as DEVICE array.
   * 
   * This function only assigns a pointer, but does not copy. It sets update
   * flags.
   *
   * @param[in] new_vals    - pointer to new values data (array of real numbers)
   * @param[in] memspace    - memory space (HOST or DEVICE) of _new_vals_
   *
   * @return 0 if successful, -1 if not.
   */  
  int matrix::Sparse::setValuesPointer(real_type* new_vals,
                                       memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    setNotUpdated();

    switch (memspace) {
      case HOST:
        if (owns_cpu_vals_ && h_val_data_) {
          out::error() << "Trying to set matrix host values, but the values already set!\n";
          out::error() << "Ignoring setValuesPointer function call ...\n";
          return 1;
        }
        h_val_data_ = new_vals;	
        h_data_updated_ = true;
        owns_cpu_vals_  = false;
        break;
      case DEVICE:
        if (owns_gpu_vals_ && d_val_data_) {
          out::error() << "Trying to set matrix device values, but the values already set!\n";
          out::error() << "Ignoring setValuesPointer function call ...\n";
          return 1;
        }
        d_val_data_ = new_vals;	
        d_data_updated_ = true;
        owns_gpu_vals_  = false;
        break;
      default:
        return -1;
    }
    return 0;
  }

} // namespace ReSolve

