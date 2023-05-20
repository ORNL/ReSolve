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
    n{n},
    m{m},
    nnz{nnz}
  {
    this->is_symmetric = false;
    this->is_expanded = true;//defaults is a normal non-symmetric fully expanded matrix
    this->nnz_expanded = nnz;

    setNotUpdated();

    //set everything to nullptr
    h_csr_p = nullptr;
    h_csr_i = nullptr;
    h_csr_x = nullptr;

    d_csr_p = nullptr;
    d_csr_i = nullptr;
    d_csr_x = nullptr;

    h_csc_p = nullptr;
    h_csc_i = nullptr;
    h_csc_x = nullptr;

    d_csc_p = nullptr;
    d_csc_i = nullptr;
    d_csc_x = nullptr;

    h_coo_rows = nullptr;
    h_coo_cols = nullptr;
    h_coo_vals = nullptr;

    d_coo_rows = nullptr;
    d_coo_cols = nullptr;
    d_coo_vals = nullptr;
  }

  resolveMatrix::resolveMatrix(resolveInt n, 
                               resolveInt m, 
                               resolveInt nnz,
                               bool symmetric,
                               bool expanded):
    n{n},
    m{m},
    nnz{nnz},
    is_symmetric{symmetric},
    is_expanded{expanded}
  {
    if (is_expanded) {
      this->nnz_expanded = nnz;
    } else {
      this->nnz_expanded = 0;
    }
    setNotUpdated();

    //set everything to nullptr
    h_csr_p = nullptr;
    h_csr_i = nullptr;
    h_csr_x = nullptr;

    d_csr_p = nullptr;
    d_csr_i = nullptr;
    d_csr_x = nullptr;

    h_csc_p = nullptr;
    h_csc_i = nullptr;
    h_csc_x = nullptr;

    d_csc_p = nullptr;
    d_csc_i = nullptr;
    d_csc_x = nullptr;

    h_coo_rows = nullptr;
    h_coo_cols = nullptr;
    h_coo_vals = nullptr;

    d_coo_rows = nullptr;
    d_coo_cols = nullptr;
    d_coo_vals = nullptr;
  }

  resolveMatrix::~resolveMatrix()
  {
  }

  void resolveMatrix::setNotUpdated()
  { 
    h_coo_updated = false;
    h_csr_updated = false;
    h_csc_updated = false;

    d_coo_updated = false;
    d_csr_updated = false;
    d_csc_updated = false;
  }

  void  resolveMatrix::copyCsr(std::string memspaceOut)
  {

    resolveInt nnz_current = nnz;
    if (is_expanded) {nnz_current = nnz_expanded;}
    if (memspaceOut == "cpu") {
      //check if we need to copy or not
      if ((d_csr_updated == true) && (h_csr_updated == false)) {
        if (h_csr_p == nullptr) {
          h_csr_p = new resolveInt[n+1];      
        }
        if (h_csr_i == nullptr) {
          h_csr_i = new resolveInt[nnz_current];      
        }
        if (h_csr_x == nullptr) {
          h_csr_x = new resolveReal[nnz_current];      
        }
        cudaMemcpy(h_csr_p, d_csr_p, (n + 1) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csr_i, d_csr_i, nnz_current * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csr_x, d_csr_x, nnz_current * sizeof(resolveReal), cudaMemcpyDeviceToHost);
        h_csr_updated = true;
      }
    }
    if (memspaceOut == "cuda") {
      if ((d_csr_updated == false) && (h_csr_updated == true)) {
        if (d_csr_p == nullptr) {
          cudaMalloc(&d_csr_p, (n + 1)*sizeof(resolveInt)); 
        }
        if (d_csr_i == nullptr) {
          cudaMalloc(&d_csr_i, nnz_current * sizeof(resolveInt)); 
        }
        if (d_csr_x == nullptr) {
          cudaMalloc(&d_csr_x, nnz_current * sizeof(resolveReal)); 
        }
        cudaMemcpy(d_csr_p, h_csr_p, (n + 1) * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csr_i, h_csr_i, nnz_current * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csr_x, h_csr_x, nnz_current * sizeof(resolveReal), cudaMemcpyHostToDevice);
        d_csr_updated = true;
      }
    }
  }

  void   resolveMatrix::copyCsc(std::string memspaceOut)
  {
  
    resolveInt nnz_current = nnz;
    if (is_expanded) {nnz_current = nnz_expanded;}
    if (memspaceOut == "cpu") {
      //check if we need to copy or not
      if ((d_csc_updated == true) && (h_csc_updated == false)) {
        if (h_csc_p == nullptr) {
          h_csc_p = new resolveInt[n+1];      
        }
        if (h_csc_i == nullptr) {
          h_csc_i = new resolveInt[nnz_current];      
        }
        if (h_csc_x == nullptr) {
          h_csc_x = new resolveReal[nnz_current];      
        }
        cudaMemcpy(h_csc_p, d_csc_p, (n + 1) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csc_i, d_csc_i, nnz_current * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csc_x, d_csc_x, nnz_current * sizeof(resolveReal), cudaMemcpyDeviceToHost);
        h_csc_updated = true;
      }
    }
    if (memspaceOut == "cuda") {
      if ((d_csc_updated == false) && (h_csc_updated == true)) {
        if (d_csc_p == nullptr) {
          cudaMalloc(&d_csc_p, (n + 1) * sizeof(resolveInt)); 
        }
        if (d_csc_i == nullptr) {
          cudaMalloc(&d_csc_i, nnz_current * sizeof(resolveInt)); 
        }
        if (d_csc_x == nullptr) {
          cudaMalloc(&d_csc_x, nnz_current * sizeof(resolveReal)); 
        }
        cudaMemcpy(d_csc_p, h_csc_p, (n + 1) * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csc_i, h_csc_i, nnz_current * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csc_x, h_csc_x, nnz_current * sizeof(resolveReal), cudaMemcpyHostToDevice);
        d_csc_updated = true;
      }
    }
  }

  void resolveMatrix::copyCoo(std::string memspaceOut)
  {
    resolveInt nnz_current = nnz;
    if (is_expanded) {nnz_current = nnz_expanded;}
    if (memspaceOut == "cpu") {
      //check if we need to copy or not
      if ((d_coo_updated == true) && (h_coo_updated == false)) {
        if (h_coo_rows == nullptr) {
          h_coo_rows = new resolveInt[nnz_current];      
        }
        if (h_coo_cols == nullptr) {
          h_coo_cols = new resolveInt[nnz_current];      
        }
        if (h_coo_vals == nullptr) {
          h_coo_vals = new resolveReal[nnz_current];      
        }
        cudaMemcpy(h_coo_rows, d_coo_rows, nnz_current * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_coo_cols, d_coo_cols, nnz_current * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_coo_vals, d_coo_vals, nnz_current * sizeof(resolveReal), cudaMemcpyDeviceToHost);
        h_coo_updated = true;
      }
    }
    if (memspaceOut == "cuda") {
      if ((d_coo_updated == false) && (h_coo_updated == true)) {
        if (d_coo_rows == nullptr) {
          cudaMalloc(&d_coo_rows, nnz_current *sizeof(resolveInt)); 
        }
        if (d_coo_cols == nullptr) {
          cudaMalloc(&d_coo_cols, nnz_current * sizeof(resolveInt)); 
        }
        if (d_coo_vals == nullptr) {
          cudaMalloc(&d_coo_vals, nnz_current * sizeof(resolveReal)); 
        }
        cudaMemcpy(d_coo_rows, h_coo_rows, nnz_current * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_coo_cols, h_coo_cols, nnz_current * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_coo_vals, h_coo_vals, nnz_current * sizeof(resolveReal), cudaMemcpyHostToDevice);
        d_coo_updated = true;
      }
    }
  }

  resolveInt resolveMatrix::getNumRows()
  {
    return this->n;
  }

  resolveInt resolveMatrix::getNumColumns()
  {
    return this->m;
  }

  resolveInt resolveMatrix::getNnz()
  {
    return this->nnz;
  }

  resolveInt resolveMatrix::getNnzExpanded()
  {
    return this->nnz_expanded;
  }

  bool resolveMatrix::symmetric()
  {
    return is_symmetric;
  }

  bool resolveMatrix::expanded()
  {
    return is_expanded;
  }

  void resolveMatrix::setSymmetric(bool symmetric)
  {
    is_symmetric = symmetric;
  }

  void resolveMatrix::setExpanded(bool expanded)
  {
    is_expanded = expanded;
  }

  void resolveMatrix::setNnzExpanded(resolveInt nnz_expanded_new)
  {
    nnz_expanded = nnz_expanded_new;
  }
  
  void resolveMatrix::setNnz(resolveInt nnz_new)
  {
    nnz = nnz_new;
  }

  resolveInt* resolveMatrix::getCsrRowPointers(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCsr("cpu");
      return this->h_csr_p;
    } else {
      if (memspace == "cuda") {
        copyCsr("cuda");
        return this->d_csr_p;
      } else {
        return nullptr;
      }
    }
  }

  resolveInt* resolveMatrix::getCsrColIndices(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCsr("cpu");
      return this->h_csr_i;
    } else {
      if (memspace == "cuda") {
        copyCsr("cuda");
        return this->d_csr_i;
      } else {
        return nullptr;
      }
    }
  }

  resolveReal* resolveMatrix::getCsrValues(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCsr("cpu");
      return this->h_csr_x;
    } else {
      if (memspace == "cuda") {
        copyCsr("cuda");
        return this->d_csr_x;
      } else {
        return nullptr;
      }
    }
  }

  resolveInt* resolveMatrix::getCscColPointers(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCsc("cpu");
      return this->h_csc_p;
    } else {
      if (memspace == "cuda") {
        copyCsc("cuda");
        return this->d_csc_p;
      } else {
        return nullptr;
      }
    }
  }

  resolveInt* resolveMatrix::getCscRowIndices(std::string memspace) 
  {
    if (memspace == "cpu") {
      copyCsc("cpu");
      return this->h_csc_i;
    } else {
      if (memspace == "cuda") {
        copyCsc("cuda");
        return this->d_csc_i;
      } else {
        return nullptr;
      }
    }
  }

  resolveReal* resolveMatrix::getCscValues(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCsc("cpu");
      return this->h_csc_x;
    } else {
      if (memspace == "cuda") {
        copyCsc("cuda");
        return this->d_csc_x;
      } else {
        return nullptr;
      }
    }
  }

  resolveInt* resolveMatrix::getCooRowIndices(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCoo("cpu");
      return this->h_coo_rows;
    } else {
      if (memspace == "cuda") {
        copyCoo("cuda");
        return this->d_coo_rows;
      } else {
        return nullptr;
      }
    }
  }

  resolveInt* resolveMatrix::getCooColIndices(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCoo("cpu");
      return this->h_coo_cols;
    } else {
      if (memspace == "cuda") {
        copyCoo("cuda");
        return this->d_coo_cols;
      } else {
        return nullptr;
      }
    }
  }

  resolveReal* resolveMatrix::getCooValues(std::string memspace)
  {
    if (memspace == "cpu") {
      copyCoo("cpu");
      return this->h_coo_vals;
    } else {
      if (memspace == "cuda") {
        copyCoo("cuda");
        return this->d_coo_vals;
      } else {
        return nullptr;
      }
    }
  }

  int resolveMatrix::setCsr(int* csr_p, int* csr_i, double* csr_x, std::string memspace)
  {

    setNotUpdated();

    if (memspace == "cpu"){
      this->h_csr_p = csr_p;
      this->h_csr_i = csr_i;
      this->h_csr_x = csr_x;	
      h_csr_updated = true;
    } else {
      if (memspace == "cuda"){ 
        this->d_csr_p = csr_p;
        this->d_csr_i = csr_i;
        this->d_csr_x = csr_x;
        d_csr_updated = true;
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
      this->h_csc_p = csc_p;
      this->h_csc_i = csc_i;
      this->h_csc_x = csc_x;
      h_csc_updated = true;
    } else {
      if (memspace == "cuda"){ 
        this->d_csc_p = csc_p;
        this->d_csc_i = csc_i;
        this->d_csc_x = csc_x;
        d_csc_updated = true;
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
      this->h_coo_rows = coo_rows;
      this->h_coo_cols = coo_cols;
      this->h_coo_vals = coo_vals;
      h_coo_updated = true;
    } else {
      if (memspace == "cuda"){ 
        this->d_coo_rows = coo_rows;
        this->d_coo_cols = coo_cols;
        this->d_coo_vals = coo_vals;
        d_coo_updated = true;
      } else {
        return -1;
      }
    }
    return 0;
  }


  resolveInt resolveMatrix::updateCsr(resolveInt* csr_p, resolveInt* csr_i, resolveReal* csr_x,  std::string memspaceIn, std::string memspaceOut)
  {
    //four cases (for now)
    resolveInt nnz_current = nnz;
    if (is_expanded) {nnz_current = nnz_expanded;}
    setNotUpdated();
    int control=-1;
    if ((memspaceIn == "cpu") && (memspaceOut == "cpu")){ control = 0;}
    if ((memspaceIn == "cpu") && (memspaceOut == "cuda")){ control = 1;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cpu")){ control = 2;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cuda")){ control = 3;}

    if (memspaceOut == "cpu") {
      //check if cpu data allocated
      if (h_csr_p == nullptr) {
        this->h_csr_p = new resolveInt[n+1];
      }
      if (h_csr_i == nullptr) {
        this->h_csr_i = new resolveInt[nnz_current];
      } 
      if (h_csr_x == nullptr) {
        this->h_csr_x = new resolveReal[nnz_current];
      }
    }

    if (memspaceOut == "cuda") {
      //check if cuda data allocated
      if (d_csr_p == nullptr) {
        cudaMalloc(&d_csr_p, (n + 1)*sizeof(resolveInt)); 
      }
      if (d_csr_i == nullptr) {
        cudaMalloc(&d_csr_i, nnz_current * sizeof(resolveInt)); 
      }
      if (d_csr_x == nullptr) {
        cudaMalloc(&d_csr_x, nnz_current * sizeof(resolveReal)); 
      }
    }

    //copy	
    switch(control)  {
      case 0: //cpu->cpu
        std::memcpy(h_csr_p, csr_p, (n + 1) * sizeof(resolveInt));
        std::memcpy(h_csr_i, csr_i, (nnz_current) * sizeof(resolveInt));
        std::memcpy(h_csr_x, csr_x, (nnz_current) * sizeof(resolveReal));
        h_csr_updated = true;
        break;
      case 2://cuda->cpu
        cudaMemcpy(h_csr_p, csr_p, (n + 1) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csr_i, csr_i, (nnz_current) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csr_x, csr_x, (nnz_current) * sizeof(resolveReal), cudaMemcpyDeviceToHost);
        h_csr_updated = true;
        break;
      case 1://cpu->cuda
        cudaMemcpy(d_csr_p, csr_p, (n + 1) * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csr_i, csr_i, (nnz_current) * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csr_x, csr_x, (nnz_current) * sizeof(resolveReal), cudaMemcpyHostToDevice);
        d_csr_updated = true;
        break;
      case 3://cuda->cuda
        cudaMemcpy(d_csr_p, csr_p, (n + 1) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_csr_i, csr_i, (nnz_current) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_csr_x, csr_x, (nnz_current) * sizeof(resolveReal), cudaMemcpyDeviceToDevice);
        d_csr_updated = true;
        break;
      default:
        return -1;
    }
    return 0;
  }

  resolveInt resolveMatrix::updateCsc(resolveInt* csc_p, resolveInt* csc_i, resolveReal* csc_x,  std::string memspaceIn, std::string memspaceOut)
  {

    resolveInt nnz_current = nnz;
    if (is_expanded) {nnz_current = nnz_expanded;}
    //four cases (for now)
    int control=-1;
    setNotUpdated();
    if ((memspaceIn == "cpu") && (memspaceOut == "cpu")){ control = 0;}
    if ((memspaceIn == "cpu") && (memspaceOut == "cuda")){ control = 1;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cpu")){ control = 2;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cuda")){ control = 3;}

    if (memspaceOut == "cpu") {
      //check if cpu data allocated
      if (h_csc_p == nullptr) {
        this->h_csc_p = new resolveInt[n+1];
      }
      if (h_csc_i == nullptr) {
        this->h_csc_i = new resolveInt[nnz_current];
      } 
      if (h_csc_x == nullptr) {
        this->h_csc_x = new resolveReal[nnz_current];
      }
    }

    if (memspaceOut == "cuda") {
      //check if cuda data allocated
      if (d_csc_p == nullptr) {
        cudaMalloc(&d_csc_p, (n + 1)*sizeof(resolveInt)); 
      }
      if (d_csc_i == nullptr) {
        cudaMalloc(&d_csc_i, nnz_current * sizeof(resolveInt)); 
      }
      if (d_csc_x == nullptr) {
        cudaMalloc(&d_csc_x, nnz_current * sizeof(resolveReal)); 
      }
    }

    switch(control)  {
      case 0: //cpu->cpu
        std::memcpy(h_csc_p, csc_p, (n + 1) * sizeof(resolveInt));
        std::memcpy(h_csc_i, csc_i, (nnz_current) * sizeof(resolveInt));
        std::memcpy(h_csc_x, csc_x, (nnz_current) * sizeof(resolveReal));
        h_csc_updated = true;
        break;
      case 2://cuda->cpu
        cudaMemcpy(h_csc_p, csc_p, (n + 1) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csc_i, csc_i, (nnz_current) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csc_x, csc_x, (nnz_current) * sizeof(resolveReal), cudaMemcpyDeviceToHost);
        h_csc_updated = true;
        break;
      case 1://cpu->cuda
        cudaMemcpy(d_csc_p, csc_p, (n + 1) * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csc_i, csc_i, (nnz_current) * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csc_x, csc_x, (nnz_current) * sizeof(resolveReal), cudaMemcpyHostToDevice);
        d_csc_updated = true;
        break;
      case 3://cuda->cuda
        cudaMemcpy(d_csc_p, csc_p, (n + 1) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_csc_i, csc_i, (nnz_current) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_csc_x, csc_x, (nnz_current) * sizeof(resolveReal), cudaMemcpyDeviceToDevice);
        d_csc_updated = true;
        break;
      default:
        return -1;
    }
    return 0;
  }

  resolveInt resolveMatrix::updateCoo(resolveInt* coo_rows, resolveInt* coo_cols, resolveReal* coo_vals,  std::string memspaceIn, std::string memspaceOut)
  {
    //four cases (for now)
    resolveInt nnz_current = nnz;
    if (is_expanded) {nnz_current = nnz_expanded;}
    setNotUpdated();
    int control=-1;
    if ((memspaceIn == "cpu") && (memspaceOut == "cpu")){ control = 0;}
    if ((memspaceIn == "cpu") && (memspaceOut == "cuda")){ control = 1;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cpu")){ control = 2;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cuda")){ control = 3;}

    if (memspaceOut == "cpu") {
      //check if cpu data allocated	
      if (h_coo_rows == nullptr) {
        this->h_coo_rows = new resolveInt[nnz_current];
      }
      if (h_coo_cols == nullptr) {
        this->h_coo_cols = new resolveInt[nnz_current];
      }
      if (h_coo_vals == nullptr) {
        this->h_coo_vals = new resolveReal[nnz_current];
      }
    }

    if (memspaceOut == "cuda") {
      //check if cuda data allocated
      if (d_coo_rows == nullptr) {
        cudaMalloc(&d_coo_rows, (nnz_current) * sizeof(resolveInt)); 
      }
      if (d_coo_cols == nullptr) {
        cudaMalloc(&d_coo_cols, nnz_current * sizeof(resolveInt)); 
      }
      if (d_coo_vals == nullptr) {
        cudaMalloc(&d_coo_vals, nnz_current * sizeof(resolveReal)); 
      }
    }

    switch(control)  {
      case 0: //cpu->cpu
        std::memcpy(h_coo_rows, coo_rows, (nnz_current) * sizeof(resolveInt));
        std::memcpy(h_coo_cols, coo_cols, (nnz_current) * sizeof(resolveInt));
        std::memcpy(h_coo_vals, coo_vals, (nnz_current) * sizeof(resolveReal));
        h_coo_updated = true;
        break;
      case 2://cuda->cpu
        cudaMemcpy(h_coo_rows, coo_rows, (nnz_current) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_coo_cols, coo_cols, (nnz_current) * sizeof(resolveInt), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_coo_vals, coo_vals, (nnz_current) * sizeof(resolveReal), cudaMemcpyDeviceToHost);
        h_coo_updated = true;
        break;
      case 1://cpu->cuda
        cudaMemcpy(d_coo_rows, coo_rows, (nnz_current) * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_coo_cols, coo_cols, (nnz_current) * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_coo_vals, coo_vals, (nnz_current) * sizeof(resolveReal), cudaMemcpyHostToDevice);
        d_coo_updated = true;
        break;
      case 3://cuda->cuda
        cudaMemcpy(d_coo_rows, coo_rows, (nnz_current) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_coo_cols, coo_cols, (nnz_current) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_coo_vals, coo_vals, (nnz_current) * sizeof(resolveReal), cudaMemcpyDeviceToDevice);
        d_coo_updated = true;
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
      if (h_csr_p != nullptr) delete [] h_csr_p;
      if (h_csr_i != nullptr) delete [] h_csr_i;
      if (h_csr_x != nullptr) delete [] h_csr_x;
    } else {
      if (memspace == "cuda"){ 
        if (d_csr_p != nullptr) cudaFree(h_csr_p);
        if (d_csr_i != nullptr) cudaFree(h_csr_i);
        if (d_csr_x != nullptr) cudaFree(h_csr_x);
      } else {
        return -1;
      }
    }
    return 0;
  }

  resolveInt resolveMatrix::destroyCsc(std::string memspace)
  {   
    if (memspace == "cpu"){  
      if (h_csc_p != nullptr) delete [] h_csc_p;
      if (h_csc_i != nullptr) delete [] h_csc_i;
      if (h_csc_x != nullptr) delete [] h_csc_x;
    } else {
      if (memspace == "cuda"){ 
        if (d_csc_p != nullptr) cudaFree(h_csc_p);
        if (d_csc_i != nullptr) cudaFree(h_csc_i);
        if (d_csc_x != nullptr) cudaFree(h_csc_x);
      } else {
        return -1;
      }
    }
    return 0;
  }

  resolveInt resolveMatrix::destroyCoo(std::string memspace)
  { 
    if (memspace == "cpu"){  
      if (h_coo_rows != nullptr) delete [] h_coo_rows;
      if (h_coo_cols != nullptr) delete [] h_coo_cols;
      if (h_coo_vals != nullptr) delete [] h_coo_vals;
    } else {
      if (memspace == "cuda"){ 
        if (d_coo_rows != nullptr) cudaFree(h_coo_rows);
        if (d_coo_cols != nullptr) cudaFree(h_coo_cols);
        if (d_coo_vals != nullptr) cudaFree(h_coo_vals);
      } else {
        return -1;
      }
    }
    return 0;
  }
}
