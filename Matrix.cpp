#include "Matrix.hpp"
#include <cuda_runtime.h>

namespace ReSolve 
{
  Matrix::Matrix()
  {
  }

  Matrix::Matrix(Int n, 
                               Int m, 
                               Int nnz):
    n_{n},
    m_{m},
    nnz_{nnz}
  {
    this->is_symmetric_ = false;
    this->is_expanded_ = true;//defaults is a normal non-symmetric fully expanded matrix
    this->nnz_expanded_ = nnz;

    setNotUpdated();

    //set everything to nullptr
    h_csr_p_ = nullptr;
    h_csr_i_ = nullptr;
    h_csr_x_ = nullptr;

    d_csr_p_ = nullptr;
    d_csr_i_ = nullptr;
    d_csr_x_ = nullptr;

    h_csc_p_ = nullptr;
    h_csc_i_ = nullptr;
    h_csc_x_ = nullptr;

    d_csc_p_ = nullptr;
    d_csc_i_ = nullptr;
    d_csc_x_ = nullptr;

    h_coo_rows_ = nullptr;
    h_coo_cols_ = nullptr;
    h_coo_vals_ = nullptr;

    d_coo_rows_ = nullptr;
    d_coo_cols_ = nullptr;
    d_coo_vals_ = nullptr;
  }

  Matrix::Matrix(Int n, 
                               Int m, 
                               Int nnz,
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
    h_csr_p_ = nullptr;
    h_csr_i_ = nullptr;
    h_csr_x_ = nullptr;

    d_csr_p_ = nullptr;
    d_csr_i_ = nullptr;
    d_csr_x_ = nullptr;

    h_csc_p_ = nullptr;
    h_csc_i_ = nullptr;
    h_csc_x_ = nullptr;

    d_csc_p_ = nullptr;
    d_csc_i_ = nullptr;
    d_csc_x_ = nullptr;

    h_coo_rows_ = nullptr;
    h_coo_cols_ = nullptr;
    h_coo_vals_ = nullptr;

    d_coo_rows_ = nullptr;
    d_coo_cols_ = nullptr;
    d_coo_vals_ = nullptr;
  }

  Matrix::~Matrix()
  {
    this->destroyCsr("cpu");
    this->destroyCsr("cuda");
    
    this->destroyCsc("cpu");
    this->destroyCsc("cuda");
  
    this->destroyCoo("cpu");
    this->destroyCoo("cuda");
  }

  void Matrix::setNotUpdated()
  { 
    h_coo_updated_ = false;
    h_csr_updated_ = false;
    h_csc_updated_ = false;

    d_coo_updated_ = false;
    d_csr_updated_ = false;
    d_csc_updated_ = false;
  }

  void  Matrix::copyCsr(std::string memspaceOut)
  {

    Int nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}

    if (memspaceOut == "cpu") {
      //check if we need to copy or not
      if ((d_csr_updated_ == true) && (h_csr_updated_ == false)) {
        if (h_csr_p_ == nullptr) {
          h_csr_p_ = new Int[n_ + 1];      
        }
        if (h_csr_i_ == nullptr) {
          h_csr_i_ = new Int[nnz_current];      
        }
        if (h_csr_x_ == nullptr) {
          h_csr_x_ = new Real[nnz_current];      
        }
        cudaMemcpy(h_csr_p_, d_csr_p_, (n_ + 1) * sizeof(Int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csr_i_, d_csr_i_, nnz_current * sizeof(Int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csr_x_, d_csr_x_, nnz_current * sizeof(Real), cudaMemcpyDeviceToHost);
        h_csr_updated_ = true;
      }
    }

    if (memspaceOut == "cuda") {
      if ((d_csr_updated_ == false) && (h_csr_updated_ == true)) {
        if (d_csr_p_ == nullptr) {
          cudaMalloc(&d_csr_p_, (n_ + 1)*sizeof(Int)); 
        }
        if (d_csr_i_ == nullptr) {
          cudaMalloc(&d_csr_i_, nnz_current * sizeof(Int)); 
        }
        if (d_csr_x_ == nullptr) {
          cudaMalloc(&d_csr_x_, nnz_current * sizeof(Real)); 
        }
        cudaMemcpy(d_csr_p_, h_csr_p_, (n_ + 1) * sizeof(Int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csr_i_, h_csr_i_, nnz_current * sizeof(Int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csr_x_, h_csr_x_, nnz_current * sizeof(Real), cudaMemcpyHostToDevice);
        d_csr_updated_ = true;
      }
    }
  }

  void   Matrix::copyCsc(std::string memspaceOut)
  {  
    Int nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
   
    if (memspaceOut == "cpu") {
      //check if we need to copy or not
      if ((d_csc_updated_ == true) && (h_csc_updated_ == false)) {
        if (h_csc_p_ == nullptr) {
          h_csc_p_ = new Int[n_ + 1];      
        }
        if (h_csc_i_ == nullptr) {
          h_csc_i_ = new Int[nnz_current];      
        }
        if (h_csc_x_ == nullptr) {
          h_csc_x_ = new Real[nnz_current];      
        }
        cudaMemcpy(h_csc_p_, d_csc_p_, (n_ + 1) * sizeof(Int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csc_i_, d_csc_i_, nnz_current * sizeof(Int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csc_x_, d_csc_x_, nnz_current * sizeof(Real), cudaMemcpyDeviceToHost);
        h_csc_updated_ = true;
      }
    }
   
    if (memspaceOut == "cuda") {
      if ((d_csc_updated_ == false) && (h_csc_updated_ == true)) {
        if (d_csc_p_ == nullptr) {
          cudaMalloc(&d_csc_p_, (n_ + 1) * sizeof(Int)); 
        }
        if (d_csc_i_ == nullptr) {
          cudaMalloc(&d_csc_i_, nnz_current * sizeof(Int)); 
        }
        if (d_csc_x_ == nullptr) {
          cudaMalloc(&d_csc_x_, nnz_current * sizeof(Real)); 
        }
        cudaMemcpy(d_csc_p_, h_csc_p_, (n_ + 1) * sizeof(Int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csc_i_, h_csc_i_, nnz_current * sizeof(Int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csc_x_, h_csc_x_, nnz_current * sizeof(Real), cudaMemcpyHostToDevice);
        d_csc_updated_ = true;
      }
    }
  }

  void Matrix::copyCoo(std::string memspaceOut)
  {
    Int nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
   
    if (memspaceOut == "cpu") {
      //check if we need to copy or not
      if ((d_coo_updated_ == true) && (h_coo_updated_ == false)) {
        if (h_coo_rows_ == nullptr) {
          h_coo_rows_ = new Int[nnz_current];      
        }
        if (h_coo_cols_ == nullptr) {
          h_coo_cols_ = new Int[nnz_current];      
        }
        if (h_coo_vals_ == nullptr) {
          h_coo_vals_ = new Real[nnz_current];      
        }
        cudaMemcpy(h_coo_rows_, d_coo_rows_, nnz_current * sizeof(Int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_coo_cols_, d_coo_cols_, nnz_current * sizeof(Int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_coo_vals_, d_coo_vals_, nnz_current * sizeof(Real), cudaMemcpyDeviceToHost);
        h_coo_updated_ = true;
      }
    }
   
    if (memspaceOut == "cuda") {
      if ((d_coo_updated_ == false) && (h_coo_updated_ == true)) {
        if (d_coo_rows_ == nullptr) {
          cudaMalloc(&d_coo_rows_, nnz_current *sizeof(Int)); 
        }
        if (d_coo_cols_ == nullptr) {
          cudaMalloc(&d_coo_cols_, nnz_current * sizeof(Int)); 
        }
        if (d_coo_vals_ == nullptr) {
          cudaMalloc(&d_coo_vals_, nnz_current * sizeof(Real)); 
        }
        cudaMemcpy(d_coo_rows_, h_coo_rows_, nnz_current * sizeof(Int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_coo_cols_, h_coo_cols_, nnz_current * sizeof(Int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_coo_vals_, h_coo_vals_, nnz_current * sizeof(Real), cudaMemcpyHostToDevice);
        d_coo_updated_ = true;
      }
    }
  }

  Int Matrix::getNumRows()
  {
    return this->n_;
  }

  Int Matrix::getNumColumns()
  {
    return this->m_;
  }

  Int Matrix::getNnz()
  {
    return this->nnz_;
  }

  Int Matrix::getNnzExpanded()
  {
    return this->nnz_expanded_;
  }

  bool Matrix::symmetric()
  {
    return is_symmetric_;
  }

  bool Matrix::expanded()
  {
    return is_expanded_;
  }

  void Matrix::setSymmetric(bool symmetric)
  {
    this->is_symmetric_ = symmetric;
  }

  void Matrix::setExpanded(bool expanded)
  {
    this->is_expanded_ = expanded;
  }

  void Matrix::setNnzExpanded(Int nnz_expanded_new)
  {
    this->nnz_expanded_ = nnz_expanded_new;
  }
  
  void Matrix::setNnz(Int nnz_new)
  {
    this->nnz_ = nnz_new;
  }

 void Matrix::setUpdated(std::string what)
 {
   if (what == "h_csr") 
   {
     h_csr_updated_ = true;
     d_csr_updated_ = false;
   }
   if (what == "d_csr") 
   {
     d_csr_updated_ = true;
     h_csr_updated_ = false;
   }
   if (what == "h_csc") 
   {
     h_csc_updated_ = true;
     d_csc_updated_ = false;
   }
   if (what == "d_csc") 
   {
     d_csc_updated_ = true;
     h_csc_updated_ = false;
   }
   if (what == "h_coo") 
   {
     h_coo_updated_ = true;
     d_coo_updated_ = false;
   }
   if (what == "d_coo") 
   {
     d_coo_updated_ = true;
     h_coo_updated_ = false;
   }
 }

  Int* Matrix::getCsrRowPointers(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCsr("cpu");
      return this->h_csr_p_;
    } else {
      if (memspace == "cuda") {
        copyCsr("cuda");
        return this->d_csr_p_;
      } else {
        return nullptr;
      }
    }
  }

  Int* Matrix::getCsrColIndices(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCsr("cpu");
      return this->h_csr_i_;
    } else {
      if (memspace == "cuda") {
        copyCsr("cuda");
        return this->d_csr_i_;
      } else {
        return nullptr;
      }
    }
  }

  Real* Matrix::getCsrValues(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCsr("cpu");
      return this->h_csr_x_;
    } else {
      if (memspace == "cuda") {
        copyCsr("cuda");
        return this->d_csr_x_;
      } else {
        return nullptr;
      }
    }
  }

  Int* Matrix::getCscColPointers(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCsc("cpu");
      return this->h_csc_p_;
    } else {
      if (memspace == "cuda") {
        copyCsc("cuda");
        return this->d_csc_p_;
      } else {
        return nullptr;
      }
    }
  }

  Int* Matrix::getCscRowIndices(std::string memspace) 
  {
    if (memspace == "cpu") {
      copyCsc("cpu");
      return this->h_csc_i_;
    } else {
      if (memspace == "cuda") {
        copyCsc("cuda");
        return this->d_csc_i_;
      } else {
        return nullptr;
      }
    }
  }

  Real* Matrix::getCscValues(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCsc("cpu");
      return this->h_csc_x_;
    } else {
      if (memspace == "cuda") {
        copyCsc("cuda");
        return this->d_csc_x_;
      } else {
        return nullptr;
      }
    }
  }

  Int* Matrix::getCooRowIndices(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCoo("cpu");
      return this->h_coo_rows_;
    } else {
      if (memspace == "cuda") {
        copyCoo("cuda");
        return this->d_coo_rows_;
      } else {
        return nullptr;
      }
    }
  }

  Int* Matrix::getCooColIndices(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCoo("cpu");
      return this->h_coo_cols_;
    } else {
      if (memspace == "cuda") {
        copyCoo("cuda");
        return this->d_coo_cols_;
      } else {
        return nullptr;
      }
    }
  }

  Real* Matrix::getCooValues(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCoo("cpu");
      return this->h_coo_vals_;
    } else {
      if (memspace == "cuda") {
        copyCoo("cuda");
        return this->d_coo_vals_;
      } else {
        return nullptr;
      }
    }
  }

  int Matrix::setCsr(int* csr_p, int* csr_i, double* csr_x, std::string memspace)
  {

    setNotUpdated();

    if (memspace == "cpu"){
      this->h_csr_p_ = csr_p;
      this->h_csr_i_ = csr_i;
      this->h_csr_x_ = csr_x;	
      h_csr_updated_ = true;
    } else {
      if (memspace == "cuda"){ 
        this->d_csr_p_ = csr_p;
        this->d_csr_i_ = csr_i;
        this->d_csr_x_ = csr_x;
        d_csr_updated_ = true;
      } else {
        return -1;
      }
    }
    return 0;
  }

  int Matrix::setCsc(int* csc_p, int* csc_i, double* csc_x, std::string memspace)
  {
    setNotUpdated();

    if (memspace == "cpu"){
      this->h_csc_p_ = csc_p;
      this->h_csc_i_ = csc_i;
      this->h_csc_x_ = csc_x;
      h_csc_updated_ = true;
    } else {
      if (memspace == "cuda"){ 
        this->d_csc_p_ = csc_p;
        this->d_csc_i_ = csc_i;
        this->d_csc_x_ = csc_x;
        d_csc_updated_ = true;
      } else {
        return -1;
      }
    }
    return 0;
  }

  int Matrix::setCoo(int* coo_rows, int* coo_cols, double* coo_vals, std::string memspace)
  {
    setNotUpdated();
    if (memspace == "cpu"){
      this->h_coo_rows_ = coo_rows;
      this->h_coo_cols_ = coo_cols;
      this->h_coo_vals_ = coo_vals;
      h_coo_updated_ = true;
    } else {
      if (memspace == "cuda"){ 
        this->d_coo_rows_ = coo_rows;
        this->d_coo_cols_ = coo_cols;
        this->d_coo_vals_ = coo_vals;
        d_coo_updated_ = true;
      } else {
        return -1;
      }
    }
    return 0;
  }


  Int Matrix::updateCsr(Int* csr_p, Int* csr_i, Real* csr_x,  std::string memspaceIn, std::string memspaceOut)
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
      if (h_csr_p_ == nullptr) {
        this->h_csr_p_ = new Int[n_ + 1];
      }
      if (h_csr_i_ == nullptr) {
        this->h_csr_i_ = new Int[nnz_current];
      } 
      if (h_csr_x_ == nullptr) {
        this->h_csr_x_ = new Real[nnz_current];
      }
    }

    if (memspaceOut == "cuda") {
      //check if cuda data allocated
      if (d_csr_p_ == nullptr) {
        cudaMalloc(&d_csr_p_, (n_ + 1) * sizeof(Int)); 
      }
      if (d_csr_i_ == nullptr) {
        cudaMalloc(&d_csr_i_, nnz_current * sizeof(Int)); 
      }
      if (d_csr_x_ == nullptr) {
        cudaMalloc(&d_csr_x_, nnz_current * sizeof(Real)); 
      }
    }

    //copy	
    switch(control)  {
      case 0: //cpu->cpu
        std::memcpy(h_csr_p_, csr_p, (n_ + 1) * sizeof(Int));
        std::memcpy(h_csr_i_, csr_i, (nnz_current) * sizeof(Int));
        std::memcpy(h_csr_x_, csr_x, (nnz_current) * sizeof(Real));
        h_csr_updated_ = true;
        break;
      case 2://cuda->cpu
        cudaMemcpy(h_csr_p_, csr_p, (n_ + 1) * sizeof(Int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csr_i_, csr_i, (nnz_current) * sizeof(Int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csr_x_, csr_x, (nnz_current) * sizeof(Real), cudaMemcpyDeviceToHost);
        h_csr_updated_ = true;
        break;
      case 1://cpu->cuda
        cudaMemcpy(d_csr_p_, csr_p, (n_ + 1) * sizeof(Int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csr_i_, csr_i, (nnz_current) * sizeof(Int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csr_x_, csr_x, (nnz_current) * sizeof(Real), cudaMemcpyHostToDevice);
        d_csr_updated_ = true;
        break;
      case 3://cuda->cuda
        cudaMemcpy(d_csr_p_, csr_p, (n_ + 1) * sizeof(Int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_csr_i_, csr_i, (nnz_current) * sizeof(Int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_csr_x_, csr_x, (nnz_current) * sizeof(Real), cudaMemcpyDeviceToDevice);
        d_csr_updated_ = true;
        break;
      default:
        return -1;
    }
    return 0;
  }

  Int Matrix::updateCsc(Int* csc_p, Int* csc_i, Real* csc_x,  std::string memspaceIn, std::string memspaceOut)
  {

    Int nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    //four cases (for now)
    int control=-1;
    setNotUpdated();
    if ((memspaceIn == "cpu") && (memspaceOut == "cpu")){ control = 0;}
    if ((memspaceIn == "cpu") && (memspaceOut == "cuda")){ control = 1;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cpu")){ control = 2;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cuda")){ control = 3;}

    if (memspaceOut == "cpu") {
      //check if cpu data allocated
      if (h_csc_p_ == nullptr) {
        this->h_csc_p_ = new Int[n_ + 1];
      }
      if (h_csc_i_ == nullptr) {
        this->h_csc_i_ = new Int[nnz_current];
      } 
      if (h_csc_x_ == nullptr) {
        this->h_csc_x_ = new Real[nnz_current];
      }
    }

    if (memspaceOut == "cuda") {
      //check if cuda data allocated
      if (d_csc_p_ == nullptr) {
        cudaMalloc(&d_csc_p_, (n_ + 1) * sizeof(Int)); 
      }
      if (d_csc_i_ == nullptr) {
        cudaMalloc(&d_csc_i_, nnz_current * sizeof(Int)); 
      }
      if (d_csc_x_ == nullptr) {
        cudaMalloc(&d_csc_x_, nnz_current * sizeof(Real)); 
      }
    }

    switch(control)  {
      case 0: //cpu->cpu
        std::memcpy(h_csc_p_, csc_p, (n_ + 1) * sizeof(Int));
        std::memcpy(h_csc_i_, csc_i, (nnz_current) * sizeof(Int));
        std::memcpy(h_csc_x_, csc_x, (nnz_current) * sizeof(Real));
        h_csc_updated_ = true;
        break;
      case 2://cuda->cpu
        cudaMemcpy(h_csc_p_, csc_p, (n_ + 1) * sizeof(Int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csc_i_, csc_i, (nnz_current) * sizeof(Int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csc_x_, csc_x, (nnz_current) * sizeof(Real), cudaMemcpyDeviceToHost);
        h_csc_updated_ = true;
        break;
      case 1://cpu->cuda
        cudaMemcpy(d_csc_p_, csc_p, (n_ + 1) * sizeof(Int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csc_i_, csc_i, (nnz_current) * sizeof(Int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csc_x_, csc_x, (nnz_current) * sizeof(Real), cudaMemcpyHostToDevice);
        d_csc_updated_ = true;
        break;
      case 3://cuda->cuda
        cudaMemcpy(d_csc_p_, csc_p, (n_ + 1) * sizeof(Int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_csc_i_, csc_i, (nnz_current) * sizeof(Int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_csc_x_, csc_x, (nnz_current) * sizeof(Real), cudaMemcpyDeviceToDevice);
        d_csc_updated_ = true;
        break;
      default:
        return -1;
    }
    return 0;
  }

  Int Matrix::updateCoo(Int* coo_rows, Int* coo_cols, Real* coo_vals,  std::string memspaceIn, std::string memspaceOut)
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
      if (h_coo_rows_ == nullptr) {
        this->h_coo_rows_ = new Int[nnz_current];
      }
      if (h_coo_cols_ == nullptr) {
        this->h_coo_cols_ = new Int[nnz_current];
      }
      if (h_coo_vals_ == nullptr) {
        this->h_coo_vals_ = new Real[nnz_current];
      }
    }

    if (memspaceOut == "cuda") {
      //check if cuda data allocated
      if (d_coo_rows_ == nullptr) {
        cudaMalloc(&d_coo_rows_, nnz_current * sizeof(Int)); 
      }
      if (d_coo_cols_ == nullptr) {
        cudaMalloc(&d_coo_cols_, nnz_current * sizeof(Int)); 
      }
      if (d_coo_vals_ == nullptr) {
        cudaMalloc(&d_coo_vals_, nnz_current * sizeof(Real)); 
      }
    }

    switch(control)  {
      case 0: //cpu->cpu
        std::memcpy(h_coo_rows_, coo_rows, (nnz_current) * sizeof(Int));
        std::memcpy(h_coo_cols_, coo_cols, (nnz_current) * sizeof(Int));
        std::memcpy(h_coo_vals_, coo_vals, (nnz_current) * sizeof(Real));
        h_coo_updated_ = true;
        break;
      case 2://cuda->cpu
        cudaMemcpy(h_coo_rows_, coo_rows, (nnz_current) * sizeof(Int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_coo_cols_, coo_cols, (nnz_current) * sizeof(Int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_coo_vals_, coo_vals, (nnz_current) * sizeof(Real), cudaMemcpyDeviceToHost);
        h_coo_updated_ = true;
        break;
      case 1://cpu->cuda
        cudaMemcpy(d_coo_rows_, coo_rows, (nnz_current) * sizeof(Int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_coo_cols_, coo_cols, (nnz_current) * sizeof(Int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_coo_vals_, coo_vals, (nnz_current) * sizeof(Real), cudaMemcpyHostToDevice);
        d_coo_updated_ = true;
        break;
      case 3://cuda->cuda
        cudaMemcpy(d_coo_rows_, coo_rows, (nnz_current) * sizeof(Int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_coo_cols_, coo_cols, (nnz_current) * sizeof(Int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_coo_vals_, coo_vals, (nnz_current) * sizeof(Real), cudaMemcpyDeviceToDevice);
        d_coo_updated_ = true;
        break;
      default:
        return -1;
    }
    return 0;
  }


  Int Matrix::updateCsr(Int* csr_p, Int* csr_i, Real* csr_x, Int new_nnz, std::string memspaceIn, std::string memspaceOut)
  {
    this->destroyCsr(memspaceOut);
    int i = this->updateCsr(csr_p, csr_i, csr_x, memspaceIn, memspaceOut);
    return i;
  }

  Int Matrix::updateCsc(Int* csc_p, Int* csc_i, Real* csc_x, Int new_nnz, std::string memspaceIn, std::string memspaceOut)
  {
    this->destroyCsc(memspaceOut);
    int i = this->updateCsc(csc_p, csc_i, csc_x, memspaceIn, memspaceOut);
    return i;
  }

  Int Matrix::updateCoo(Int* coo_rows, Int* coo_cols, Real* coo_vals, Int new_nnz, std::string memspaceIn, std::string memspaceOut)
  {
    this->destroyCoo(memspaceOut);
    int i = this->updateCoo(coo_rows, coo_cols, coo_vals, memspaceIn, memspaceOut);
    return i;
  }

  Int Matrix::destroyCsr(std::string memspace)
  {
    if (memspace == "cpu"){  
      if (h_csr_p_ != nullptr) delete [] h_csr_p_;
      if (h_csr_i_ != nullptr) delete [] h_csr_i_;
      if (h_csr_x_ != nullptr) delete [] h_csr_x_;
    } else {
      if (memspace == "cuda"){ 
        if (d_csr_p_ != nullptr) cudaFree(d_csr_p_);
        if (d_csr_i_ != nullptr) cudaFree(d_csr_i_);
        if (d_csr_x_ != nullptr) cudaFree(d_csr_x_);
      } else {
        return -1;
      }
    }
    return 0;
  }

  Int Matrix::destroyCsc(std::string memspace)
  {   
    if (memspace == "cpu"){  
      if (h_csc_p_ != nullptr) delete [] h_csc_p_;
      if (h_csc_i_ != nullptr) delete [] h_csc_i_;
      if (h_csc_x_ != nullptr) delete [] h_csc_x_;
    } else {
      if (memspace == "cuda"){ 
        if (d_csc_p_ != nullptr) cudaFree(d_csc_p_);
        if (d_csc_i_ != nullptr) cudaFree(d_csc_i_);
        if (d_csc_x_ != nullptr) cudaFree(d_csc_x_);
      } else {
        return -1;
      }
    }
    return 0;
  }

  Int Matrix::destroyCoo(std::string memspace)
  { 
    if (memspace == "cpu"){  
      if (h_coo_rows_ != nullptr) delete [] h_coo_rows_;
      if (h_coo_cols_ != nullptr) delete [] h_coo_cols_;
      if (h_coo_vals_ != nullptr) delete [] h_coo_vals_;
    } else {
      if (memspace == "cuda"){ 
        if (d_coo_rows_ != nullptr) cudaFree(d_coo_rows_);
        if (d_coo_cols_ != nullptr) cudaFree(d_coo_cols_);
        if (d_coo_vals_ != nullptr) cudaFree(d_coo_vals_);
      } else {
        return -1;
      }
    }
    return 0;
  }
 
  void Matrix::allocateCsr(std::string memspace)
  {
    Int nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    destroyCsr(memspace);//just in case

    if (memspace == "cpu") {
      this->h_csr_p_ = new Int[n_ + 1];
      this->h_csr_i_ = new Int[nnz_current];
      this->h_csr_x_ = new Real[nnz_current];
    }

    if (memspace == "cuda") {
      cudaMalloc(&d_csr_p_, (n_ + 1) * sizeof(Int)); 
      cudaMalloc(&d_csr_i_, nnz_current * sizeof(Int)); 
      cudaMalloc(&d_csr_x_, nnz_current * sizeof(Real)); 
    }
  }


  void Matrix::allocateCsc(std::string memspace)
  {
    Int nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    destroyCsc(memspace);//just in case

    if (memspace == "cpu") {
      this->h_csc_p_ = new Int[n_ + 1];
      this->h_csc_i_ = new Int[nnz_current];
      this->h_csc_x_ = new Real[nnz_current];
    }

    if (memspace == "cuda") {
      cudaMalloc(&d_csc_p_, (n_ + 1) * sizeof(Int)); 
      cudaMalloc(&d_csc_i_, nnz_current * sizeof(Int)); 
      cudaMalloc(&d_csc_x_, nnz_current * sizeof(Real)); 
    }
  }
  
  void Matrix::allocateCoo(std::string memspace)
  {
    Int nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    destroyCoo(memspace);//just in case

    if (memspace == "cpu") {
      this->h_coo_rows_ = new Int[nnz_current];
      this->h_coo_cols_ = new Int[nnz_current];
      this->h_coo_vals_ = new Real[nnz_current];
    }

    if (memspace == "cuda") {
      cudaMalloc(&d_coo_rows_, nnz_current * sizeof(Int)); 
      cudaMalloc(&d_coo_cols_, nnz_current * sizeof(Int)); 
      cudaMalloc(&d_coo_vals_, nnz_current * sizeof(Real)); 
    }
  }
}
