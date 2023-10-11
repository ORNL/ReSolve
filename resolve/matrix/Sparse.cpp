#include <cstring>  // <-- includes memcpy
#include "Sparse.hpp"
#include <resolve/memoryUtils.hpp>

namespace ReSolve { namespace matrix {

  /** 
   * @brief empty constructor that does absolutely nothing        
   */
  Sparse::Sparse()
  {
  }

  /** 
   * @brief basic constructor that sets matrix dimensions and nnz. Note: it does not allocate any storage. 
   *
   * @param n number of rows
   * @param m number of columns
   * @param nnz number of non-zeroes      
   */
  Sparse::Sparse(index_type n, 
                 index_type m, 
                 index_type nnz):
    n_{n},
    m_{m},
    nnz_{nnz}
  {
    this->is_symmetric_ = false;
    this->is_expanded_ = true; //default is a normal non-symmetric fully expanded matrix
    this->nnz_expanded_ = nnz;

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
   * @brief constructor that sets matrix dimensions, nnz, informs if the matrix is symmetric and expanded. Note: it does not allocate any storage. 
   *
   * @param n number of rows
   * @param m number of columns
   * @param nnz number of non-zeroes      
   * @param symmetric boolean variable - 1 if the matrix is symmetric, 0 otherwise. 
   * @param expanded boolean variable - 1 if the matrix is expanded, 0 otherwise. Note: non-symmetric matries are always considered to be expanded
   */
  Sparse::Sparse(index_type n, 
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

    owns_cpu_data_ = false;
    owns_cpu_vals_ = false;
    
    owns_gpu_data_ = false;
    owns_gpu_vals_ = false;
  }

  /** 
   * @brief destructor. If the matrix owns its own data, the data is deleted. Simply clears the object otherwise. 
   */
  Sparse::~Sparse()
  {
    this->destroyMatrixData("cpu");
    this->destroyMatrixData("cuda");
  }

  /** 
   * @brief sets both cpu and gpu data to be "not updated". 
   */
  void Sparse::setNotUpdated()
  {
    h_data_updated_ = false;
    d_data_updated_ = false; 
  }
  
  /** 
   * @brief get number of rows. 
   *
   * @return number of rows
  */
  index_type Sparse::getNumRows()
  {
    return this->n_;
  }

  /** 
   * @brief get number of columns. 
   *
   * @return number of rows
   */
  index_type Sparse::getNumColumns()
  {
    return this->m_;
  }

  /** 
   * @brief get number of non-zeroes. 
   *
   * @return number of non-zeross
   */
  index_type Sparse::getNnz()
  {
    return this->nnz_;
  }

  /** 
   * @brief get number of non-zeroes in the expanded matrix. 
   *
   * @return number of non-zeros in expanded matrix
   */
  index_type Sparse::getNnzExpanded()
  {
    return this->nnz_expanded_;
  }

  /** 
   * @bried get matrix symmetry property
   *
   * @return  1 if the matrix is symmtric and 0 otherwise. 
   */
  bool Sparse::symmetric()
  {
    return is_symmetric_;
  }

  /** 
   * @brief get the info whether the matrix is expanded or not.
   *
   * @return returns 1 if the matrix is expanded and 0 otherwise. 
   */
  bool Sparse::expanded()
  {
    return is_expanded_;
  }

  /** 
   * @brief set matrix symmetry property
   *
   * @param symmetric use 1 to set matrix to symmetric and 0 to set matrix to non-symmetric. 
   */
  void Sparse::setSymmetric(bool symmetric)
  {
    this->is_symmetric_ = symmetric;
  }

  /** 
   * @brief set whether the matrix is expanded or not.
   *
   * @param expanded use 1 to set matrix to expanded and 0 to set matrix to non-expanded. 
   */
  void Sparse::setExpanded(bool expanded)
  {
    this->is_expanded_ = expanded;
  }

  /** 
   * @brief set number of non-zeroes in expanded matrix,
   *
   * @param nnz_expanded_new number of non-zeroes in expanded matrix. 
   */
  void Sparse::setNnzExpanded(index_type nnz_expanded_new)
  {
    this->nnz_expanded_ = nnz_expanded_new;
  }

  /** 
   * @brief set number of non-zeroes (in non-expanded matrix),
   *
   * @param nnz_new number of non-zeroes in non-expanded matrix. 
   */
  void Sparse::setNnz(index_type nnz_new)
  {
    this->nnz_ = nnz_new;
  }

  /** 
   * @brief set the "updated" parameter for matrix data,
   *
   * @param what use "cpu" to set CPU data to be updated and "cuda" for GPU data. 
   *
   * @return 0 if succesful, -1 otherwise
  */
  int Sparse::setUpdated(std::string what)
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

  /** 
   * @brief set the matrix data (without copying, just set the pointers),
   *
   * @param row_data pointer to row data
   * @param col_data pointer to column data
   * @param val_data pointer to value data.
   * @param memspace memory space of the pointers given ("cpu" or "cuda")
   *
   * @return 0 if succesful, -1 otherwise
   */
  int Sparse::setMatrixData(index_type* row_data, index_type* col_data, real_type* val_data, std::string memspace)
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

  /** 
   * @brief destroy matrix data (if the matrix owns its data), do nothing otherwise,
   *
   * @param memspace memory space of data to be destroyed ("cpu" or "cuda")
   *
   * @return 0 if succesful, -1 otherwise
   */
  int Sparse::destroyMatrixData(std::string memspace)
  { 
    if (memspace == "cpu"){  
      if (owns_cpu_data_) {
        delete [] h_row_data_;
        delete [] h_col_data_;
      }
      if (owns_cpu_vals_) {
        delete [] h_val_data_;
      }
    } else {
      if (memspace == "cuda"){ 
        if (owns_gpu_data_) {
          deleteOnDevice(d_row_data_);
          deleteOnDevice(d_col_data_);
        }
        if (owns_gpu_vals_) {
          deleteOnDevice(d_val_data_);
        }
      } else {
        return -1;
      }
    }
    return 0;
  }

  /** 
   * @brief update matrix values (with copying). Note: if the data in memory space of memspaceOut does not exist, if will be allocated.
   *
   * @param new_vals pointer to value array
   * @param memspaceIn memory space of data arriving ("cpu" or "cuda")
   * @param memspaceOut memory space of data to be updated ("cpu" or "cuda")
   *
   * @return 0 if succesful, -1 otherwise
   */
  int Sparse::updateValues(real_type* new_vals, std::string memspaceIn, std::string memspaceOut)
  {
 
    index_type nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    //four cases (for now)
    setNotUpdated();
    int control=-1;
    if ((memspaceIn == "cpu") && (memspaceOut == "cpu")){ control = 0;}
    if ((memspaceIn == "cpu") && (memspaceOut == "cuda")){ control = 1;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cpu")){ control = 2;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cuda")){ control = 3;}
   
    if (memspaceOut == "cpu") {
      //check if cpu data allocated
      if (h_val_data_ == nullptr) {
        this->h_val_data_ = new real_type[nnz_current];
      }
    }

    if (memspaceOut == "cuda") {
      //check if cuda data allocated
      if (d_val_data_ == nullptr) {
        allocateArrayOnDevice(&d_val_data_, nnz_current); 
      }
    }

    switch(control)  {
      case 0: //cpu->cpu
        std::memcpy(h_val_data_, new_vals, (nnz_current) * sizeof(real_type));
        h_data_updated_ = true;
        owns_cpu_vals_ = true;
        break;
      case 2://cuda->cpu
        copyArrayDeviceToHost(h_val_data_, new_vals, nnz_current);
        h_data_updated_ = true;
        owns_cpu_vals_ = true;
        break;
      case 1://cpu->cuda
        copyArrayHostToDevice(d_val_data_, new_vals, nnz_current);
        d_data_updated_ = true;
        owns_gpu_vals_ = true;
        break;
      case 3://cuda->cuda
        copyArrayDeviceToDevice(d_val_data_, new_vals, nnz_current);
        d_data_updated_ = true;
        owns_gpu_vals_ = true;
        break;
      default:
        return -1;
    }
    return 0;
  }

  /** 
   * @brief update matrix values pointer (without copying).
   *
   * @param new_vals pointer to value array
   * @param memspace memory space of data ("cpu" or "cuda")
   *
   * @return 0 if succesful, -1 otherwise
   */
  int Sparse::setNewValues(real_type* new_vals, std::string memspace)
  {

    setNotUpdated();

    if (memspace == "cpu"){
      this->h_val_data_ = new_vals;	
      h_data_updated_ = true;
    } else {
      if (memspace == "cuda"){ 
        this->d_val_data_ = new_vals;	
        d_data_updated_ = true;
      } else {
        return -1;
      }
    }
    return 0;
  }

}} // namespace ReSolve::matrix
