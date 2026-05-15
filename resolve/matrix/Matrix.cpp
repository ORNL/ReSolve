#include "Matrix.hpp"

namespace ReSolve {
  /**
   * @brief empty constructor that does absolutely nothing
   */
  matrix::Matrix::Matrix()
  {
  }

  /**
   * @brief basic constructor
   *
   * @param[in] n   - number of rows
   * @param[in] m   - number of columns
   * @param[in] nnz - number of non-zeros
   */
  matrix::Matrix::Matrix(index_type n,
                         index_type m,
                         index_type nnz)
    : n_{n},
      m_{m},
      nnz_{nnz}
  {
  }

  /**
   * @brief get number of matrix rows
   *
   * @return number of matrix rows.
   */
  index_type matrix::Matrix::getNumRows()
  {
    return this->n_;
  }

  /**
   * @brief get number of matrix columns
   *
   * @return number of matrix columns.
   */
  index_type matrix::Matrix::getNumColumns()
  {
    return this->m_;
  }

  /**
   * @brief get number of non-zeros in the matrix.
   *
   * @return number of non-zeros.
   */
  index_type matrix::Matrix::getNnz()
  {
    return this->nnz_;
  }

  /**
   * @brief Set number of non-zeros.
   *
   * @param[in] nnz_new - new number of non-zeros
   */
  void matrix::Matrix::setNnz(index_type nnz_new)
  {
    this->nnz_ = nnz_new;
  }

  /**
  * @brief Loads or reloads pointer to the matrix handler for matrix operations.
  * @param[in] matrixHandler - New matrix handler
  */
  void matrix::Matrix::addMatrixHandler(MatrixHandler* matrixHandler)
  {
    this->matrixHandler_ = matrixHandler;
  }

  /**
  * @brief Implementation missing in MatrixHandler.
  */
  int matrix::Matrix::transpose(Matrix* At, memory::MemorySpace memspace)
  {
    return -1;
  }

  /**
  * @brief Implementation missing in MatrixHandler.
  */
  int matrix::Matrix::leftScale(vector_type* diag, memory::MemorySpace memspace)
  {
    return -1;
  }

  /**
  * @brief Implementation missing in MatrixHandler.
  */
  int matrix::Matrix::rightScale(vector_type* diag, memory::MemorySpace memspace)
  {
    return -1;
  }

  /**
  * @brief Implementation missing in MatrixHandler.
  */
  int matrix::Matrix::addConst(real_type alpha, memory::MemorySpace memspace)
  {
    return -1;
  }

  /**
  * @brief Implementation missing in MatrixHandler.
  */
  int matrix::Matrix::matvec(
    vector_type*        vec_x,
    vector_type*        vec_result,
    const real_type*    alpha,
    const real_type*    beta,
    memory::MemorySpace memspace)
  {
    return -1;
  }

  /**
  * @brief Implementation missing in MatrixHandler.
  */
  int matrix::Matrix::matrixInfNorm(real_type* norm, memory::MemorySpace memspace)
  {
    return -1;
  }

}
