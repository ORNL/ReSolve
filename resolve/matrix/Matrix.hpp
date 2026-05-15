#pragma once

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/Vector.hpp>

namespace ReSolve
{
  using vector_type = vector::Vector;

  namespace matrix
  {
    /**
     * @brief This class implements the matrix interface.
     *
     * This is a parent class for all matrix classes, including Sparse and, in the future,
     * HyKKT matrices. This also provides an interface for matrix operations implemented
     * in MatrixHandler.
     */
    class Matrix
    {
    public:
      Matrix();
      Matrix(index_type n, index_type m, index_type nnz);

      index_type getNumRows();
      index_type getNumColumns();
      index_type getNnz();

      void        setNnz(index_type nnz_new); // for resetting when removing duplicates
      virtual int setUpdated(memory::MemorySpace memspace) = 0;

      void addMatrixHandler(MatrixHandler* matrixHandler);

      virtual int  allocateMatrixData(memory::MemorySpace memspace)        = 0;
      virtual int  destroyMatrixData(memory::MemorySpace memspace)         = 0;
      virtual void print(std::ostream& file_out, index_type indexing_base) = 0;

      // Matrix operations. They need to be properly implemented in MatrixHandler as well for the specific derived class.
      virtual int transpose(Matrix* At, memory::MemorySpace memspace);
      virtual int leftScale(vector_type* diag, memory::MemorySpace memspace);
      virtual int rightScale(vector_type* diag, memory::MemorySpace memspace);
      virtual int addConst(real_type alpha, memory::MemorySpace memspace);
      virtual int matvec(
          vector_type*        vec_x,
          vector_type*        vec_result,
          const real_type*    alpha,
          const real_type*    beta,
          memory::MemorySpace memspace);
      virtual int matrixInfNorm(real_type* norm, memory::MemorySpace memspace);

    protected:
      index_type n_{0};   ///< number of rows
      index_type m_{0};   ///< number of columns
      index_type nnz_{0}; ///< number of non-zeros

      MatrixHandler* matrixHandler_{nullptr};
    };
  } // namespace matrix
} // namespace ReSolve
