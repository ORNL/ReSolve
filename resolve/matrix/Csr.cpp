#include <cstring>  // <-- includes memcpy
#include <resolve/memoryUtils.hpp>
#include "Csr.hpp"

namespace ReSolve 
{
  /** 
   * @brief empty constructor that does absolutely nothing        
   */
  matrix::Csr::Csr()
  {
  }

  /** 
   * @brief basic constructor that sets matrix dimensions and nnz. Note: it does not allocate any storage. 
   *
   * @param n number of rows
   * @param m number of columns
   * @param nnz number of non-zeroes      
   */
  matrix::Csr::Csr(index_type n, index_type m, index_type nnz) : Sparse(n, m, nnz)
  {
  }
  
  /** 
   * @brief constructor that sets matrix dimensions, nnz, informs if the matrix is symmetric and expanded. Note: it does not allocate any storage. 
   *
   * @param n number of rows
   * @param m number of columns
   * @param nnz number of non-zeroes      
   * @param symmetric boolean variable - 1 if the matrix is symmetric, 0 otherwise. 
   * @param expanded boolean variable - 1 if the matrix is expanded, 0 otherwise. Note: non-symmetric matries are always considered to be expanded
   */
  matrix::Csr::Csr(index_type n, 
                   index_type m, 
                   index_type nnz,
                   bool symmetric,
                   bool expanded) : Sparse(n, m, nnz, symmetric, expanded)
  {
  }

  /** 
   * @brief destructor. If the matrix owns its own data, the data is deleted. Simply clears the object otherwise. 
   */
  matrix::Csr::~Csr()
  {
  }

  /** 
   * @brief get CSR row pointers. Note: if the data in given memory space does not exist, it will be allocated and copied first.
   *
   * @param memspace use "cpu" to get pointer to CPU data  and "cuda" for GPU data. 
   *
   * @return pointer to the data in memory space given or nullpointer if the memory space parameter is wrong.
  */
  index_type* matrix::Csr::getRowData(std::string memspace)
  {
    if (memspace == "cpu") {
      copyData("cpu");
      return this->h_row_data_;
    } else {
      if (memspace == "cuda") {
        copyData("cuda");
        return this->d_row_data_;
      } else {
        return nullptr;
      }
    }
  }

  /** 
   * @brief get CSR column indices. Note: if the data in given memory space does not exist, it will be allocated and copied first.
   *
   * @param memspace use "cpu" to get pointer to CPU data  and "cuda" for GPU data. 
   *
   * @return pointer to the data in memory space given or nullpointer if the memory space parameter is wrong.
  */
  index_type* matrix::Csr::getColData(std::string memspace)
  {
    if (memspace == "cpu") {
      copyData("cpu");
      return this->h_col_data_;
    } else {
      if (memspace == "cuda") {
        copyData("cuda");
        return this->d_col_data_;
      } else {
        return nullptr;
      }
    }
  }

  /** 
   * @brief get CSR values. Note: if the data in given memory space does not exist, it will be allocated and copied first.
   *
   * @param memspace use "cpu" to get pointer to CPU data  and "cuda" for GPU data. 
   *
   * @return pointer to the data in memory space given or nullpointer if the memory space parameter is wrong.
  */
  real_type* matrix::Csr::getValues(std::string memspace)
  {
    if (memspace == "cpu") {
      copyData("cpu");
      return this->h_val_data_;
    } else {
      if (memspace == "cuda") {
        copyData("cuda");
        return this->d_val_data_;
      } else {
        return nullptr;
      }
    }
  }

  /** 
   * @brief updade CSR data. Note: if the data in given memory space "memspaceOut" does not exist, it will be allocated first.
   *
   * @param row_data CSR row pointers
   * @param col_data CSR column indices
   * @param val_data CSR values
   * @param memspaceIn memory space of the input data. Use "cpu" or "cuda", as appropriate. 
   * @param memspaceOut memory space of the matrix data to be updated. Use "cpu" or "cuda", as appropriate. 
   *
   * @return 0 if succesful and -1  if the memory space parameter is wrong.
  */
  int matrix::Csr::updateData(index_type* row_data, index_type* col_data, real_type* val_data, std::string memspaceIn, std::string memspaceOut)
  {
    //four cases (for now)
    index_type nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    setNotUpdated();
    int control = -1;
    if ((memspaceIn == "cpu") && (memspaceOut == "cpu")){ control = 0;}
    if ((memspaceIn == "cpu") && (memspaceOut == "cuda")){ control = 1;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cpu")){ control = 2;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cuda")){ control = 3;}

    if (memspaceOut == "cpu") {
      //check if cpu data allocated
      if (h_row_data_ == nullptr) {
        this->h_row_data_ = new index_type[n_ + 1];
      }
      if (h_col_data_ == nullptr) {
        this->h_col_data_ = new index_type[nnz_current];
      } 
      if (h_val_data_ == nullptr) {
        this->h_val_data_ = new real_type[nnz_current];
      }
    }

    if (memspaceOut == "cuda") {
      //check if cuda data allocated
      if (d_row_data_ == nullptr) {
        allocateArrayOnDevice(&d_row_data_, n_ + 1); 
      }
      if (d_col_data_ == nullptr) {
        allocateArrayOnDevice(&d_col_data_, nnz_current);
      }
      if (d_val_data_ == nullptr) {
        allocateArrayOnDevice(&d_val_data_, nnz_current); 
      }
    }


    //copy	
    switch(control)  {
      case 0: //cpu->cpu
        std::memcpy(h_row_data_, row_data, (n_ + 1) * sizeof(index_type));
        std::memcpy(h_col_data_, col_data, (nnz_current) * sizeof(index_type));
        std::memcpy(h_val_data_, val_data, (nnz_current) * sizeof(real_type));
        h_data_updated_ = true;
        owns_cpu_data_ = true;
        owns_cpu_vals_ = true;
        break;
      case 2://cuda->cpu
        copyArrayDeviceToHost(h_row_data_, row_data,      n_ + 1);
        copyArrayDeviceToHost(h_col_data_, col_data, nnz_current);
        copyArrayDeviceToHost(h_val_data_, val_data, nnz_current);
        h_data_updated_ = true;
        owns_cpu_data_ = true;
        owns_cpu_vals_ = true;
        break;
      case 1://cpu->cuda
        copyArrayHostToDevice(d_row_data_, row_data,      n_ + 1);
        copyArrayHostToDevice(d_col_data_, col_data, nnz_current);
        copyArrayHostToDevice(d_val_data_, val_data, nnz_current);
        d_data_updated_ = true;
        owns_gpu_data_ = true;
        owns_gpu_vals_ = true;
        break;
      case 3://cuda->cuda
        copyArrayDeviceToDevice(d_row_data_, row_data,      n_ + 1);
        copyArrayDeviceToDevice(d_col_data_, col_data, nnz_current);
        copyArrayDeviceToDevice(d_val_data_, val_data, nnz_current);
        d_data_updated_ = true;
        owns_gpu_data_ = true;
        owns_gpu_vals_ = true;
        break;
      default:
        return -1;
    }
    return 0;
  } 

  /** 
   * @brief updade CSR data with new nnz. Note: all existing data is destroyed first. New data is allocated only in memory space "memspaceOut".
   *
   * @param row_data CSR row pointers
   * @param col_data CSR column indices
   * @param val_data CSR values
   * @param new_nnz new number of non-zeros
   * @param memspaceIn memory space of the input data. Use "cpu" or "cuda", as appropriate. 
   * @param memspaceOut memory space of the matrix data to be updated. Use "cpu" or "cuda", as appropriate. 
   *
   * @return 0 if succesful and -1  if the one or both memory space parameters are wrong.
  */
  int matrix::Csr::updateData(index_type* row_data, index_type* col_data, real_type* val_data, index_type new_nnz, std::string memspaceIn, std::string memspaceOut)
  {
    this->destroyMatrixData(memspaceOut);
    this->nnz_ = new_nnz;
    int i = this->updateData(row_data, col_data, val_data, memspaceIn, memspaceOut);
    return i;
  } 

  /** 
   * @brief allocate CSR data in memory space provided. Note: all existing data is destroyed first.
   *
   * @param memspace memory space of the matrix data to be allocated. Use "cpu" or "cuda", as appropriate. 
   *
   * @return 0 if succesful and -1  if memory space parameters is wrong.
  */
  int matrix::Csr::allocateMatrixData(std::string memspace)
  {
    index_type nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    destroyMatrixData(memspace);//just in case

    if (memspace == "cpu") {
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

    if (memspace == "cuda") {
      allocateArrayOnDevice(&d_row_data_,      n_ + 1); 
      allocateArrayOnDevice(&d_col_data_, nnz_current); 
      allocateArrayOnDevice(&d_val_data_, nnz_current); 
      owns_gpu_data_ = true;
      owns_gpu_vals_ = true;
      return 0;  
    }
    return -1;
  }

  /** 
   * @brief (internally) copy the matrix data to memory space indicated by "memspaceOut". Note: this function allocatedthe target data if it does not exist.
   *
   * @param memspaceOut memory space of the matrix data to be copied to. Use "cpu" or "cuda", as appropriate. 
   *
   * @return 0 if succesful and -1  if memory space parameters is wrong.
  */
  int matrix::Csr::copyData(std::string memspaceOut)
  {
    index_type nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}

    if (memspaceOut == "cpu") {
      //check if we need to copy or not
      if ((d_data_updated_ == true) && (h_data_updated_ == false)) {
        if (h_row_data_ == nullptr) {
          h_row_data_ = new index_type[n_ + 1];      
        }
        if (h_col_data_ == nullptr) {
          h_col_data_ = new index_type[nnz_current];      
        }
        if (h_val_data_ == nullptr) {
          h_val_data_ = new real_type[nnz_current];      
        }
        copyArrayDeviceToHost(h_row_data_, d_row_data_,      n_ + 1);
        copyArrayDeviceToHost(h_col_data_, d_col_data_, nnz_current);
        copyArrayDeviceToHost(h_val_data_, d_val_data_, nnz_current);
        h_data_updated_ = true;
        owns_cpu_data_ = true;
        owns_cpu_vals_ = true;
      }
      return 0;
    }

    if (memspaceOut == "cuda") {
      if ((d_data_updated_ == false) && (h_data_updated_ == true)) {
        if (d_row_data_ == nullptr) {
          allocateArrayOnDevice(&d_row_data_, n_ + 1); 
        }
        if (d_col_data_ == nullptr) {
          allocateArrayOnDevice(&d_col_data_, nnz_current); 
        }
        if (d_val_data_ == nullptr) {
          allocateArrayOnDevice(&d_val_data_, nnz_current); 
        }
        copyArrayHostToDevice(d_row_data_, h_row_data_,      n_ + 1);
        copyArrayHostToDevice(d_col_data_, h_col_data_, nnz_current);
        copyArrayHostToDevice(d_val_data_, h_val_data_, nnz_current);
        d_data_updated_ = true;
        owns_gpu_data_ = true;
        owns_gpu_vals_ = true;
      }
      return 0;
    }
  return -1;  
}

} // namespace ReSolve 

