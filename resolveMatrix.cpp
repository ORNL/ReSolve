#include "resolveMatrix.hpp"
#include <cuda_runtime.h>

namespace ReSolve {

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
  }

  resolveMatrix::~resolveMatrix()
  {
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

  resolveInt* resolveMatrix::getCsrRowPointers(std::string memspace)
  {
    if (memspace == "cpu") {
      return this->h_csr_p;
    } else {
      if (memspace == "cuda") {
        return this->d_csr_p;
      } else {
        return nullptr;
      }
    }
  }

  resolveInt* resolveMatrix::getCsrColIndices(std::string memspace)
  {
    if (memspace == "cpu") {
      return this->h_csr_i;
    } else {
      if (memspace == "cuda") {
        return this->d_csr_i;
      } else {
        return nullptr;
      }
    }
  }

  resolveReal* resolveMatrix::getCsrValues(std::string memspace)
  {
    if (memspace == "cpu") {
      return this->h_csr_x;
    } else {
      if (memspace == "cuda") {
        return this->d_csr_x;
      } else {
        return nullptr;
      }
    }
  }

  resolveInt* resolveMatrix::getCscColPointers(std::string memspace)
  {
    if (memspace == "cpu") {
      return this->h_csc_p;
    } else {
      if (memspace == "cuda") {
        return this->d_csc_p;
      } else {
        return nullptr;
      }
    }
  }

  resolveInt* resolveMatrix::getCscRowIndices(std::string memspace) 
  {
    if (memspace == "cpu") {
      return this->h_csc_i;
    } else {
      if (memspace == "cuda") {
        return this->d_csc_i;
      } else {
        return nullptr;
      }
    }
  }

  resolveReal* resolveMatrix::getCscValues(std::string memspace)
  {
    if (memspace == "cpu") {
      return this->h_csc_x;
    } else {
      if (memspace == "cuda") {
        return this->d_csc_x;
      } else {
        return nullptr;
      }
    }
  }

  resolveInt* resolveMatrix::getCooRowIndices(std::string memspace)
  {
    if (memspace == "cpu") {
      return this->h_coo_rows;
    } else {
      if (memspace == "cuda") {
        return this->d_coo_rows;
      } else {
        return nullptr;
      }
    }
  }

  resolveInt* resolveMatrix::getCooColIndices(std::string memspace)
  {
    if (memspace == "cpu") {
      return this->h_coo_cols;
    } else {
      if (memspace == "cuda") {
        return this->d_coo_cols;
      } else {
        return nullptr;
      }
    }
  }

  resolveReal* resolveMatrix::getCooValues(std::string memspace)
  {
    if (memspace == "cpu") {
      return this->h_coo_vals;
    } else {
      if (memspace == "cuda") {
        return this->d_coo_vals;
      } else {
        return nullptr;
      }
    }
  }

  resolveInt resolveMatrix::setCsr(resolveInt* csr_p, resolveInt* csr_i, resolveReal* csr_x, resolveInt new_nnz_expanded, std::string memspaceIn, std::string memspaceOut)
  {

    if (new_nnz_expanded != nnz_expanded) {
      nnz_expanded = new_nnz_expanded;
      nnz = nnz_expanded;
      this->destroyCsr(memspaceIn);
    }
    if ((memspaceIn == "cpu") && (memspaceOut == "cpu")){ // we update cpu data with cpu data
      if (h_csr_p == nullptr) {
        this->h_csr_p = csr_p;
      } else {
        std::memcpy(h_csr_p, csr_p, (n + 1) * sizeof(resolveInt));
      }
      if (h_csr_i == nullptr) {
        this->h_csr_i = csr_i;
      } else {
        std::memcpy(h_csr_i, csr_i, (nnz_expanded) * sizeof(resolveInt));
      }
      if (h_csr_x == nullptr) {
        this->h_csr_x = csr_x;
      } else {
        std::memcpy(h_csr_x, csr_x, (nnz_expanded) * sizeof(resolveReal));
      }
    } else {
      if ((memspaceIn == "cuda") && (memspaceOut == "cuda")) { 
        if (d_csr_p == nullptr) {
          this->d_csr_p = csr_p;
        } else {
          cudaMemcpy(d_csr_p, csr_p, (n + 1) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
        }
        if (d_csr_i == nullptr) {
          this->d_csr_i = csr_i;
        } else {
          cudaMemcpy(d_csr_i, csr_i, (nnz_expanded) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
        }
        if (d_csr_x == nullptr) {
          this->d_csr_x = csr_x;
        } else {
          cudaMemcpy(d_csr_x, csr_x, (nnz_expanded) * sizeof(resolveReal), cudaMemcpyDeviceToDevice);
        }
      } else {
        if ((memspaceIn == "cpu") && (memspaceOut == "cuda")) { 

          if (d_csr_p == nullptr) {
            cudaMalloc(&d_csr_p, (n + 1)*sizeof(resolveInt)); 
          }
          cudaMemcpy(d_csr_p, csr_p, (n + 1) * sizeof(resolveInt), cudaMemcpyHostToDevice);

          if (d_csr_i == nullptr) {
            cudaMalloc(&d_csr_i, nnz_expanded * sizeof(resolveInt)); 
          }
          cudaMemcpy(d_csr_i, csr_i, (nnz_expanded) * sizeof(resolveInt), cudaMemcpyHostToDevice);

          if (d_csr_x == nullptr) {
            cudaMalloc(&d_csr_x, nnz_expanded * sizeof(resolveReal)); 
          }
          cudaMemcpy(d_csr_x, csr_x, (nnz_expanded) * sizeof(resolveReal), cudaMemcpyHostToDevice);

        } else {  

          if ((memspaceIn == "cuda") && (memspaceOut == "cpu")) { 

            if (h_csr_p == nullptr) {
              h_csr_p = new resolveInt[n+1];
            }
            cudaMemcpy(h_csr_p, csr_p, (n + 1) * sizeof(resolveInt), cudaMemcpyDeviceToHost);

            if (h_csr_i == nullptr) {
              h_csr_i = new resolveInt[nnz_expanded];
            }
            cudaMemcpy(h_csr_i, csr_i, (nnz_expanded) * sizeof(resolveInt), cudaMemcpyDeviceToHost);

            if (h_csr_x == nullptr) {
              h_csr_x = new resolveReal[nnz_expanded];
            }
            cudaMemcpy(h_csr_x, csr_x, (nnz_expanded) * sizeof(resolveReal), cudaMemcpyDeviceToHost);

          } else {
            return -1;
          }
        }

      }
    }

      return 0;
  }

    resolveInt resolveMatrix::setCsc(resolveInt* csc_p, resolveInt* csc_i, resolveReal* csc_x, resolveInt new_nnz_expanded, std::string memspace)
    {
      if (memspace == "cpu"){
        if (new_nnz_expanded != nnz_expanded) {
          this->destroyCsc("cpu");
          nnz_expanded = new_nnz_expanded;
          nnz = nnz_expanded;
        }
        if (h_csc_p == nullptr) {
          this->h_csc_p = csc_p;
        } else {
          std::memcpy(h_csc_p, csc_p, (n + 1) * sizeof(resolveInt));
        }
        if (h_csc_i == nullptr) {
          this->h_csc_i = csc_i;
        } else {
          std::memcpy(h_csc_i, csc_i, (nnz_expanded) * sizeof(resolveInt));
        }
        if (h_csc_x == nullptr) {
          this->h_csc_x = csc_x;
        } else {
          std::memcpy(h_csc_x, csc_x, (nnz_expanded) * sizeof(resolveReal));
        }
      } else {
        if (memspace == "cuda"){ 
          if (new_nnz_expanded != nnz_expanded) {
            this->destroyCsc("cuda");
            nnz_expanded = new_nnz_expanded;
            nnz = nnz_expanded;
          }
          if (d_csc_p == nullptr) {
            this->d_csc_p = csc_p;
          } else {
            cudaMemcpy(d_csc_p, csc_p, (n + 1) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
          }
          if (d_csc_i == nullptr) {
            this->d_csc_i = csc_i;
          } else {
            cudaMemcpy(d_csc_i, csc_i, (nnz_expanded) * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
          }
          if (d_csc_x == nullptr) {
            this->d_csc_x = csc_x;
          } else {
            cudaMemcpy(d_csc_x, csc_x, (nnz_expanded) * sizeof(resolveReal), cudaMemcpyDeviceToDevice);
          }
        } else {
          return -1;
        }

      }
      return 0;
    }

    resolveInt resolveMatrix::setCoo(resolveInt* coo_rows, resolveInt* coo_cols, resolveReal* coo_vals, resolveInt new_nnz_expanded, std::string memspace)
    {
      if (memspace == "cpu"){
        if (new_nnz_expanded != nnz_expanded) {
          this->destroyCoo("cpu");
          nnz_expanded = new_nnz_expanded;
          nnz = nnz_expanded;
        }
        if (h_coo_rows == nullptr) {
          this->h_coo_rows = coo_rows;
        } else {
          std::memcpy(h_coo_rows, coo_rows, n * sizeof(resolveInt));
        }
        if (h_coo_cols == nullptr) {
          this->h_coo_cols = coo_cols;
        } else {
          std::memcpy(h_coo_cols, coo_cols, n * sizeof(resolveInt));
        }
        if (h_coo_vals == nullptr) {
          this->h_coo_vals = coo_vals;
        } else {
          std::memcpy(h_coo_vals, coo_vals, n * sizeof(resolveReal));
        }
      } else {
        if (memspace == "cuda"){ 
          if (new_nnz_expanded != nnz_expanded) {
            this->destroyCoo("cuda");
            nnz_expanded = new_nnz_expanded;
            nnz = nnz_expanded;
          }
          if (d_coo_rows == nullptr) {
            this->d_coo_rows = coo_rows;
          } else {
            cudaMemcpy(d_coo_rows, coo_rows, n * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
          }
          if (d_coo_cols == nullptr) {
            this->d_coo_cols = coo_cols;
          } else {
            cudaMemcpy(d_coo_cols, coo_cols, n * sizeof(resolveInt), cudaMemcpyDeviceToDevice);
          }
          if (d_csc_x == nullptr) {
            this->d_coo_vals = coo_vals;
          } else {
            cudaMemcpy(d_coo_vals, coo_vals, n * sizeof(resolveReal), cudaMemcpyDeviceToDevice);
          }
        } else {
          return -1;
        }

      }
      return 0;
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
