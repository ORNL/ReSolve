#include "resolveMatrix.hpp"
#include <cuda_runtime.h>

namespace ReSolve 
{
  resolveMatrix::resolveMatrix()
  {
  }

  resolveMatrix::resolveMatrix(resolveInt n, 
                               resolveInt m, 
                               resolveInt nnz):
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

  resolveMatrix::resolveMatrix(resolveInt n, 
                               resolveInt m, 
                               resolveInt nnz,
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

  resolveMatrix::~resolveMatrix()
  {
    this->destroyCsr("cpu");
    this->destroyCsr("cuda");
    
    this->destroyCsc("cpu");
    this->destroyCsc("cuda");
  
    this->destroyCoo("cpu");
    this->destroyCoo("cuda");
  }

  void resolveMatrix::setNotUpdated()
  { 
    h_coo_updated_ = false;
    h_csr_updated_ = false;
    h_csc_updated_ = false;

    d_coo_updated_ = false;
    d_csr_updated_ = false;
    d_csc_updated_ = false;
  }

  void  resolveMatrix::copyCsr(std::string memspaceOut)
  {

    resolveInt nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}

    if (memspaceOut == "cpu") {
      //check if we need to copy or not
      if ((d_csr_updated_ == true) && (h_csr_updated_ == false)) {
        if (h_csr_p_ == nullptr) {
          h_csr_p_ = new resolveInt[n_ + 1];      
        }
        if (h_csr_i_ == nullptr) {
          h_csr_i_ = new resolveInt[nnz_current];      
        }
        if (h_csr_x_ == nullptr) {
          h_csr_x_ = new resolveReal[nnz_current];      
        }
        cudaMemcpy(h_csr_p_, d_csr_p_, (n_ + 1) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csr_i_, d_csr_i_, nnz_current * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csr_x_, d_csr_x_, nnz_current * sizeof(resolveReal), cudaMemcpyDeviceToHost);
        h_csr_updated_ = true;
      }
    }

    if (memspaceOut == "cuda") {
      if ((d_csr_updated_ == false) && (h_csr_updated_ == true)) {
        if (d_csr_p_ == nullptr) {
          cudaMalloc(&d_csr_p_, (n_ + 1)*sizeof(resolveInt)); 
        }
        if (d_csr_i_ == nullptr) {
          cudaMalloc(&d_csr_i_, nnz_current * sizeof(resolveInt)); 
        }
        if (d_csr_x_ == nullptr) {
          cudaMalloc(&d_csr_x_, nnz_current * sizeof(resolveReal)); 
        }
        cudaMemcpy(d_csr_p_, h_csr_p_, (n_ + 1) * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csr_i_, h_csr_i_, nnz_current * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csr_x_, h_csr_x_, nnz_current * sizeof(resolveReal), cudaMemcpyHostToDevice);
        d_csr_updated_ = true;
      }
    }
  }

  void   resolveMatrix::copyCsc(std::string memspaceOut)
  {  
    resolveInt nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
   
    if (memspaceOut == "cpu") {
      //check if we need to copy or not
      if ((d_csc_updated_ == true) && (h_csc_updated_ == false)) {
        if (h_csc_p_ == nullptr) {
          h_csc_p_ = new resolveInt[n_ + 1];      
        }
        if (h_csc_i_ == nullptr) {
          h_csc_i_ = new resolveInt[nnz_current];      
        }
        if (h_csc_x_ == nullptr) {
          h_csc_x_ = new resolveReal[nnz_current];      
        }
        cudaMemcpy(h_csc_p_, d_csc_p_, (n_ + 1) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csc_i_, d_csc_i_, nnz_current * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csc_x_, d_csc_x_, nnz_current * sizeof(resolveReal), cudaMemcpyDeviceToHost);
        h_csc_updated_ = true;
      }
    }
   
    if (memspaceOut == "cuda") {
      if ((d_csc_updated_ == false) && (h_csc_updated_ == true)) {
        if (d_csc_p_ == nullptr) {
          cudaMalloc(&d_csc_p_, (n_ + 1) * sizeof(resolveInt)); 
        }
        if (d_csc_i_ == nullptr) {
          cudaMalloc(&d_csc_i_, nnz_current * sizeof(resolveInt)); 
        }
        if (d_csc_x_ == nullptr) {
          cudaMalloc(&d_csc_x_, nnz_current * sizeof(resolveReal)); 
        }
        cudaMemcpy(d_csc_p_, h_csc_p_, (n_ + 1) * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csc_i_, h_csc_i_, nnz_current * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csc_x_, h_csc_x_, nnz_current * sizeof(resolveReal), cudaMemcpyHostToDevice);
        d_csc_updated_ = true;
      }
    }
  }

  void resolveMatrix::copyCoo(std::string memspaceOut)
  {
    resolveInt nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
   
    if (memspaceOut == "cpu") {
      //check if we need to copy or not
      if ((d_coo_updated_ == true) && (h_coo_updated_ == false)) {
        if (h_coo_rows_ == nullptr) {
          h_coo_rows_ = new resolveInt[nnz_current];      
        }
        if (h_coo_cols_ == nullptr) {
          h_coo_cols_ = new resolveInt[nnz_current];      
        }
        if (h_coo_vals_ == nullptr) {
          h_coo_vals_ = new resolveReal[nnz_current];      
        }
        cudaMemcpy(h_coo_rows_, d_coo_rows_, nnz_current * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_coo_cols_, d_coo_cols_, nnz_current * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_coo_vals_, d_coo_vals_, nnz_current * sizeof(resolveReal), cudaMemcpyDeviceToHost);
        h_coo_updated_ = true;
      }
    }
   
    if (memspaceOut == "cuda") {
      if ((d_coo_updated_ == false) && (h_coo_updated_ == true)) {
        if (d_coo_rows_ == nullptr) {
          cudaMalloc(&d_coo_rows_, nnz_current *sizeof(resolveInt)); 
        }
        if (d_coo_cols_ == nullptr) {
          cudaMalloc(&d_coo_cols_, nnz_current * sizeof(resolveInt)); 
        }
        if (d_coo_vals_ == nullptr) {
          cudaMalloc(&d_coo_vals_, nnz_current * sizeof(resolveReal)); 
        }
        cudaMemcpy(d_coo_rows_, h_coo_rows_, nnz_current * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_coo_cols_, h_coo_cols_, nnz_current * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_coo_vals_, h_coo_vals_, nnz_current * sizeof(resolveReal), cudaMemcpyHostToDevice);
        d_coo_updated_ = true;
      }
    }
  }

  resolveInt resolveMatrix::getNumRows()
  {
    return this->n_;
  }

  resolveInt resolveMatrix::getNumColumns()
  {
    return this->m_;
  }

  resolveInt resolveMatrix::getNnz()
  {
    return this->nnz_;
  }

  resolveInt resolveMatrix::getNnzExpanded()
  {
    return this->nnz_expanded_;
  }

  bool resolveMatrix::symmetric()
  {
    return is_symmetric_;
  }

  bool resolveMatrix::expanded()
  {
    return is_expanded_;
  }

  void resolveMatrix::setSymmetric(bool symmetric)
  {
    this->is_symmetric_ = symmetric;
  }

  void resolveMatrix::setExpanded(bool expanded)
  {
    this->is_expanded_ = expanded;
  }

  void resolveMatrix::setNnzExpanded(resolveInt nnz_expanded_new)
  {
    this->nnz_expanded_ = nnz_expanded_new;
  }
  
  void resolveMatrix::setNnz(resolveInt nnz_new)
  {
    this->nnz_ = nnz_new;
  }

 void resolveMatrix::setUpdated(std::string what)
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

  resolveInt* resolveMatrix::getCsrRowPointers(std::string memspace)
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

  resolveInt* resolveMatrix::getCsrColIndices(std::string memspace)
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

  resolveReal* resolveMatrix::getCsrValues(std::string memspace)
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

  resolveInt* resolveMatrix::getCscColPointers(std::string memspace)
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

  resolveInt* resolveMatrix::getCscRowIndices(std::string memspace) 
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

  resolveReal* resolveMatrix::getCscValues(std::string memspace)
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

  resolveInt* resolveMatrix::getCooRowIndices(std::string memspace)
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

  resolveInt* resolveMatrix::getCooColIndices(std::string memspace)
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

  resolveReal* resolveMatrix::getCooValues(std::string memspace)
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

  int resolveMatrix::setCsr(int* csr_p, int* csr_i, double* csr_x, std::string memspace)
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

  int resolveMatrix::setCsc(int* csc_p, int* csc_i, double* csc_x, std::string memspace)
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

  int resolveMatrix::setCoo(int* coo_rows, int* coo_cols, double* coo_vals, std::string memspace)
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


  resolveInt resolveMatrix::updateCsr(resolveInt* csr_p, resolveInt* csr_i, resolveReal* csr_x,  std::string memspaceIn, std::string memspaceOut)
  {
    //four cases (for now)
    resolveInt nnz_current = nnz_;
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
        this->h_csr_p_ = new resolveInt[n_ + 1];
      }
      if (h_csr_i_ == nullptr) {
        this->h_csr_i_ = new resolveInt[nnz_current];
      } 
      if (h_csr_x_ == nullptr) {
        this->h_csr_x_ = new resolveReal[nnz_current];
      }
    }

    if (memspaceOut == "cuda") {
      //check if cuda data allocated
      if (d_csr_p_ == nullptr) {
        cudaMalloc(&d_csr_p_, (n_ + 1) * sizeof(resolveInt)); 
      }
      if (d_csr_i_ == nullptr) {
        cudaMalloc(&d_csr_i_, nnz_current * sizeof(resolveInt)); 
      }
      if (d_csr_x_ == nullptr) {
        cudaMalloc(&d_csr_x_, nnz_current * sizeof(resolveReal)); 
      }
    }

    //copy	
    switch(control)  {
      case 0: //cpu->cpu
        std::memcpy(h_csr_p_, csr_p, (n_ + 1) * sizeof(resolveInt));
        std::memcpy(h_csr_i_, csr_i, (nnz_current) * sizeof(resolveInt));
        std::memcpy(h_csr_x_, csr_x, (nnz_current) * sizeof(resolveReal));
        h_csr_updated_ = true;
        break;
      case 2://cuda->cpu
        cudaMemcpy(h_csr_p_, csr_p, (n_ + 1) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csr_i_, csr_i, (nnz_current) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csr_x_, csr_x, (nnz_current) * sizeof(resolveReal), cudaMemcpyDeviceToHost);
        h_csr_updated_ = true;
        break;
      case 1://cpu->cuda
        cudaMemcpy(d_csr_p_, csr_p, (n_ + 1) * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csr_i_, csr_i, (nnz_current) * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csr_x_, csr_x, (nnz_current) * sizeof(resolveReal), cudaMemcpyHostToDevice);
        d_csr_updated_ = true;
        break;
      case 3://cuda->cuda
        cudaMemcpy(d_csr_p_, csr_p, (n_ + 1) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_csr_i_, csr_i, (nnz_current) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_csr_x_, csr_x, (nnz_current) * sizeof(resolveReal), cudaMemcpyDeviceToDevice);
        d_csr_updated_ = true;
        break;
      default:
        return -1;
    }
    return 0;
  }

  resolveInt resolveMatrix::updateCsc(resolveInt* csc_p, resolveInt* csc_i, resolveReal* csc_x,  std::string memspaceIn, std::string memspaceOut)
  {

    resolveInt nnz_current = nnz_;
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
        this->h_csc_p_ = new resolveInt[n_ + 1];
      }
      if (h_csc_i_ == nullptr) {
        this->h_csc_i_ = new resolveInt[nnz_current];
      } 
      if (h_csc_x_ == nullptr) {
        this->h_csc_x_ = new resolveReal[nnz_current];
      }
    }

    if (memspaceOut == "cuda") {
      //check if cuda data allocated
      if (d_csc_p_ == nullptr) {
        cudaMalloc(&d_csc_p_, (n_ + 1) * sizeof(resolveInt)); 
      }
      if (d_csc_i_ == nullptr) {
        cudaMalloc(&d_csc_i_, nnz_current * sizeof(resolveInt)); 
      }
      if (d_csc_x_ == nullptr) {
        cudaMalloc(&d_csc_x_, nnz_current * sizeof(resolveReal)); 
      }
    }

    switch(control)  {
      case 0: //cpu->cpu
        std::memcpy(h_csc_p_, csc_p, (n_ + 1) * sizeof(resolveInt));
        std::memcpy(h_csc_i_, csc_i, (nnz_current) * sizeof(resolveInt));
        std::memcpy(h_csc_x_, csc_x, (nnz_current) * sizeof(resolveReal));
        h_csc_updated_ = true;
        break;
      case 2://cuda->cpu
        cudaMemcpy(h_csc_p_, csc_p, (n_ + 1) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csc_i_, csc_i, (nnz_current) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csc_x_, csc_x, (nnz_current) * sizeof(resolveReal), cudaMemcpyDeviceToHost);
        h_csc_updated_ = true;
        break;
      case 1://cpu->cuda
        cudaMemcpy(d_csc_p_, csc_p, (n_ + 1) * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csc_i_, csc_i, (nnz_current) * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csc_x_, csc_x, (nnz_current) * sizeof(resolveReal), cudaMemcpyHostToDevice);
        d_csc_updated_ = true;
        break;
      case 3://cuda->cuda
        cudaMemcpy(d_csc_p_, csc_p, (n_ + 1) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_csc_i_, csc_i, (nnz_current) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_csc_x_, csc_x, (nnz_current) * sizeof(resolveReal), cudaMemcpyDeviceToDevice);
        d_csc_updated_ = true;
        break;
      default:
        return -1;
    }
    return 0;
  }

  resolveInt resolveMatrix::updateCoo(resolveInt* coo_rows, resolveInt* coo_cols, resolveReal* coo_vals,  std::string memspaceIn, std::string memspaceOut)
  {
    //four cases (for now)
    resolveInt nnz_current = nnz_;
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
        this->h_coo_rows_ = new resolveInt[nnz_current];
      }
      if (h_coo_cols_ == nullptr) {
        this->h_coo_cols_ = new resolveInt[nnz_current];
      }
      if (h_coo_vals_ == nullptr) {
        this->h_coo_vals_ = new resolveReal[nnz_current];
      }
    }

    if (memspaceOut == "cuda") {
      //check if cuda data allocated
      if (d_coo_rows_ == nullptr) {
        cudaMalloc(&d_coo_rows_, nnz_current * sizeof(resolveInt)); 
      }
      if (d_coo_cols_ == nullptr) {
        cudaMalloc(&d_coo_cols_, nnz_current * sizeof(resolveInt)); 
      }
      if (d_coo_vals_ == nullptr) {
        cudaMalloc(&d_coo_vals_, nnz_current * sizeof(resolveReal)); 
      }
    }

    switch(control)  {
      case 0: //cpu->cpu
        std::memcpy(h_coo_rows_, coo_rows, (nnz_current) * sizeof(resolveInt));
        std::memcpy(h_coo_cols_, coo_cols, (nnz_current) * sizeof(resolveInt));
        std::memcpy(h_coo_vals_, coo_vals, (nnz_current) * sizeof(resolveReal));
        h_coo_updated_ = true;
        break;
      case 2://cuda->cpu
        cudaMemcpy(h_coo_rows_, coo_rows, (nnz_current) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_coo_cols_, coo_cols, (nnz_current) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_coo_vals_, coo_vals, (nnz_current) * sizeof(resolveReal), cudaMemcpyDeviceToHost);
        h_coo_updated_ = true;
        break;
      case 1://cpu->cuda
        cudaMemcpy(d_coo_rows_, coo_rows, (nnz_current) * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_coo_cols_, coo_cols, (nnz_current) * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_coo_vals_, coo_vals, (nnz_current) * sizeof(resolveReal), cudaMemcpyHostToDevice);
        d_coo_updated_ = true;
        break;
      case 3://cuda->cuda
        cudaMemcpy(d_coo_rows_, coo_rows, (nnz_current) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_coo_cols_, coo_cols, (nnz_current) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_coo_vals_, coo_vals, (nnz_current) * sizeof(resolveReal), cudaMemcpyDeviceToDevice);
        d_coo_updated_ = true;
        break;
      default:
        return -1;
    }
    return 0;
  }


  resolveInt resolveMatrix::updateCsr(resolveInt* csr_p, resolveInt* csr_i, resolveReal* csr_x, resolveInt new_nnz, std::string memspaceIn, std::string memspaceOut)
  {
    this->destroyCsr(memspaceOut);
    int i = this->updateCsr(csr_p, csr_i, csr_x, memspaceIn, memspaceOut);
    return i;
  }

  resolveInt resolveMatrix::updateCsc(resolveInt* csc_p, resolveInt* csc_i, resolveReal* csc_x, resolveInt new_nnz, std::string memspaceIn, std::string memspaceOut)
  {
    this->destroyCsc(memspaceOut);
    int i = this->updateCsc(csc_p, csc_i, csc_x, memspaceIn, memspaceOut);
    return i;
  }

  resolveInt resolveMatrix::updateCoo(resolveInt* coo_rows, resolveInt* coo_cols, resolveReal* coo_vals, resolveInt new_nnz, std::string memspaceIn, std::string memspaceOut)
  {
    this->destroyCoo(memspaceOut);
    int i = this->updateCoo(coo_rows, coo_cols, coo_vals, memspaceIn, memspaceOut);
    return i;
  }

  resolveInt resolveMatrix::destroyCsr(std::string memspace)
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

  resolveInt resolveMatrix::destroyCsc(std::string memspace)
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

  resolveInt resolveMatrix::destroyCoo(std::string memspace)
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
 
  void resolveMatrix::allocateCsr(std::string memspace)
  {
    resolveInt nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    destroyCsr(memspace);//just in case

    if (memspace == "cpu") {
      this->h_csr_p_ = new resolveInt[n_ + 1];
      this->h_csr_i_ = new resolveInt[nnz_current];
      this->h_csr_x_ = new resolveReal[nnz_current];
    }

    if (memspace == "cuda") {
      cudaMalloc(&d_csr_p_, (n_ + 1) * sizeof(resolveInt)); 
      cudaMalloc(&d_csr_i_, nnz_current * sizeof(resolveInt)); 
      cudaMalloc(&d_csr_x_, nnz_current * sizeof(resolveReal)); 
    }
  }


  void resolveMatrix::allocateCsc(std::string memspace)
  {
    resolveInt nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    destroyCsc(memspace);//just in case

    if (memspace == "cpu") {
      this->h_csc_p_ = new resolveInt[n_ + 1];
      this->h_csc_i_ = new resolveInt[nnz_current];
      this->h_csc_x_ = new resolveReal[nnz_current];
    }

    if (memspace == "cuda") {
      cudaMalloc(&d_csc_p_, (n_ + 1) * sizeof(resolveInt)); 
      cudaMalloc(&d_csc_i_, nnz_current * sizeof(resolveInt)); 
      cudaMalloc(&d_csc_x_, nnz_current * sizeof(resolveReal)); 
    }
  }
  
  void resolveMatrix::allocateCoo(std::string memspace)
  {
    resolveInt nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    destroyCoo(memspace);//just in case

    if (memspace == "cpu") {
      this->h_coo_rows_ = new resolveInt[nnz_current];
      this->h_coo_cols_ = new resolveInt[nnz_current];
      this->h_coo_vals_ = new resolveReal[nnz_current];
    }

    if (memspace == "cuda") {
      cudaMalloc(&d_coo_rows_, nnz_current * sizeof(resolveInt)); 
      cudaMalloc(&d_coo_cols_, nnz_current * sizeof(resolveInt)); 
      cudaMalloc(&d_coo_vals_, nnz_current * sizeof(resolveReal)); 
    }
  }
}
