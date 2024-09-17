#include <algorithm>
#include <list>
#include <resolve/Common.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include "Utilities.hpp"

namespace ReSolve
{
  using out = io::Logger;


  namespace matrix
  {
    /**
     * @brief 
     * 
     * @param A_coo - Input COO matrix without duplicates sorted in row-major order
     * @param A_csr - Output CSR matrix
     * @param memspace - memory space in the output matrix where the data is copied
     * @return int
     * 
     * @pre A_coo and A_csr matrix sizes must match.
     */
    int coo2csr_simple(matrix::Coo* A_coo, matrix::Csr* A_csr, memory::MemorySpace memspace)
    {
      if ((A_coo->getNumRows()    != A_csr->getNumRows())    ||
          (A_coo->getNumColumns() != A_csr->getNumColumns()) ||
          (A_coo->getNnz()        != A_csr->getNnz())        ||
          (A_coo->symmetric()     != A_csr->symmetric())     ||
          (A_coo->expanded()      != A_csr->expanded())) {
        io::Logger::error() << "COO and CSR matrix are not of same size or type.\n";
        return 1;
      }
      index_type n = A_coo->getNumRows();
      index_type nnz = A_coo->getNnz();
      /* const */ index_type* rows_coo = A_coo->getRowData(memory::HOST);
      /* const */ index_type* cols_coo = A_coo->getColData(memory::HOST);
      /* const */ real_type*  vals_coo = A_coo->getValues(memory::HOST);

      index_type* row_csr = new index_type[n + 1];

      row_csr[0] = 0;
      index_type i_csr = 0;
      for (index_type i = 1; i < nnz; ++i) {
        if (rows_coo[i] != rows_coo[i - 1]) {
          i_csr++;
          row_csr[i_csr] = i;
        }
      }
      row_csr[n] = nnz;

      A_csr->updateData(row_csr, cols_coo, vals_coo, memory::HOST, memspace);

      delete [] row_csr;
      
      return 0;
    }

  }
}