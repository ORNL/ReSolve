#include <cstring>  // <-- includes memcpy

#include "Sparse.hpp"

namespace ReSolve { namespace matrix {

  Sparse::Sparse()
  {
  }

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

  Sparse::~Sparse()
  {
    this->destroyMatrixData("cpu");
    this->destroyMatrixData("cuda");
  }

  void Sparse::setNotUpdated()
  {
    h_data_updated_ = false;
    d_data_updated_ = false; 
  }
  
  index_type Sparse::getNumRows()
  {
    return this->n_;
  }

  index_type Sparse::getNumColumns()
  {
    return this->m_;
  }

  index_type Sparse::getNnz()
  {
    return this->nnz_;
  }

  index_type Sparse::getNnzExpanded()
  {
    return this->nnz_expanded_;
  }

  bool Sparse::symmetric()
  {
    return is_symmetric_;
  }

  bool Sparse::expanded()
  {
    return is_expanded_;
  }

  void Sparse::setSymmetric(bool symmetric)
  {
    this->is_symmetric_ = symmetric;
  }

  void Sparse::setExpanded(bool expanded)
  {
    this->is_expanded_ = expanded;
  }

  void Sparse::setNnzExpanded(index_type nnz_expanded_new)
  {
    this->nnz_expanded_ = nnz_expanded_new;
  }

  void Sparse::setNnz(index_type nnz_new)
  {
    this->nnz_ = nnz_new;
  }

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
          mem_.deleteOnDevice(d_row_data_);
          mem_.deleteOnDevice(d_col_data_);
        }
        if (owns_gpu_vals_) {
          mem_.deleteOnDevice(d_val_data_);
        }
      } else {
        return -1;
      }
    }
    return 0;
  }

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
        mem_.allocateArrayOnDevice(&d_val_data_, nnz_current); 
      }
    }

    switch(control)  {
      case 0: //cpu->cpu
        mem_.copyArrayHostToHost(h_val_data_, new_vals, nnz_current);
        h_data_updated_ = true;
        owns_cpu_vals_ = true;
        break;
      case 2://cuda->cpu
        mem_.copyArrayDeviceToHost(h_val_data_, new_vals, nnz_current);
        h_data_updated_ = true;
        owns_cpu_vals_ = true;
        break;
      case 1://cpu->cuda
        mem_.copyArrayHostToDevice(d_val_data_, new_vals, nnz_current);
        d_data_updated_ = true;
        owns_gpu_vals_ = true;
        break;
      case 3://cuda->cuda
        mem_.copyArrayDeviceToDevice(d_val_data_, new_vals, nnz_current);
        d_data_updated_ = true;
        owns_gpu_vals_ = true;
        break;
      default:
        return -1;
    }
    return 0;
  }

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
