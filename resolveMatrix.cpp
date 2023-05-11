#include "resolveMatrix.hpp"
namespace ReSolve {

  resolveMatrix::resolveMatrix()
  {
  }

  resolveMatrix::resolveMatrix(int n, int m, int nnz):
    n{n},
    m{m},
    nnz{nnz}
  {
  }

  resolveMatrix::~resolveMatrix()
  {
  }

  int resolveMatrix::getNumRows()
  {
    return this->n;
  }

  int resolveMatrix::getNumColumns()
  {
    return this->m;
  }

  int resolveMatrix::getNnz()
  {
    return this->nnz;
  }

  int* resolveMatrix::getCsrRowPointers(std::string memspace)
  {
    if (memspace == "cpu") {
      return this->h_csr_p;
    } else {
      if (memspace == "gpu") {
        return this->d_csr_p;
      } else {
        return nullptr;
      }
    }
  }

  int* resolveMatrix::getCsrColIndices(std::string memspace)
  {
    if (memspace == "cpu") {
      return this->h_csr_i;
    } else {
      if (memspace == "gpu") {
        return this->d_csr_i;
      } else {
        return nullptr;
      }
    }
  }

  double* resolveMatrix::getCsrValues(std::string memspace)
  {
    if (memspace == "cpu") {
      return this->h_csr_x;
    } else {
      if (memspace == "gpu") {
        return this->d_csr_x;
      } else {
        return nullptr;
      }
    }
  }

  int* resolveMatrix::getCscColPointers(std::string memspace)
  {
    if (memspace == "cpu") {
      return this->h_csc_p;
    } else {
      if (memspace == "gpu") {
        return this->d_csc_p;
      } else {
        return nullptr;
      }
    }
  }

  int* resolveMatrix::getCscRowIndices(std::string memspace) 
  {
    if (memspace == "cpu") {
      return this->h_csc_i;
    } else {
      if (memspace == "gpu") {
        return this->d_csc_i;
      } else {
        return nullptr;
      }
    }
  }

  double* resolveMatrix::getCscValues(std::string memspace)
  {
    if (memspace == "cpu") {
      return this->h_csc_x;
    } else {
      if (memspace == "gpu") {
        return this->d_csc_x;
      } else {
        return nullptr;
      }
    }
  }

  int* resolveMatrix::getCooRowIndices(std::string memspace)
  {
    if (memspace == "cpu") {
      return this->h_coo_rows;
    } else {
      if (memspace == "gpu") {
        return this->d_coo_rows;
      } else {
        return nullptr;
      }
    }
  }

  int* resolveMatrix::getCooColIndices(std::string memspace)
  {
    if (memspace == "cpu") {
      return this->h_coo_cols;
    } else {
      if (memspace == "gpu") {
        return this->d_coo_cols;
      } else {
        return nullptr;
      }
    }
  }

  double* resolveMatrix::getCooValues(std::string memspace)
  {
    if (memspace == "cpu") {
      return this->h_coo_vals;
    } else {
      if (memspace == "gpu") {
        return this->d_coo_vals;
      } else {
        return nullptr;
      }
    }
  }

  int resolveMatrix::setCsr(int* csr_p, int* csr_i, double* csr_x, std::string memspace)
  {
    if (memspace == "cpu"){
      this->h_csr_p = csr_p;
      this->h_csr_i = csr_i;
      this->h_csr_x = csr_x;
    } else {
      if (memspace == "gpu"){ 
        this->d_csr_p = csr_p;
        this->d_csr_i = csr_i;
        this->d_csr_x = csr_x;
      } else {
        return -1;
      }

    }
    return 0;
  }

  int resolveMatrix::setCsc(int* csc_p, int* csc_i, double* csc_x, std::string memspace)
  {
    if (memspace == "cpu"){
      this->h_csc_p = csc_p;
      this->h_csc_i = csc_i;
      this->h_csc_x = csc_x;
    } else {
      if (memspace == "gpu"){ 
        this->d_csc_p = csc_p;
        this->d_csc_i = csc_i;
        this->d_csc_x = csc_x;
      } else {
        return -1;
      }

    }
    return 0;
  }

  int resolveMatrix::setCoo(int* coo_rows, int* coo_cols, double* coo_vals, std::string memspace)
  {
    if (memspace == "cpu"){
      this->h_coo_rows = coo_rows;
      this->h_coo_cols = coo_cols;
      this->h_coo_vals = coo_vals;
    } else {
      if (memspace == "gpu"){ 
        this->d_coo_rows = coo_rows;
        this->d_coo_cols = coo_cols;
        this->d_coo_vals = coo_vals;
      } else {
        return -1;
      }
    }
    return 0;
  }

}
