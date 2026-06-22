#include "VectorHandlerCpu.hpp"

#include <cassert>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandlerImpl.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

namespace ReSolve
{
  using out = io::Logger;

  /**
   * @brief empty constructor that does absolutely nothing
   */
  VectorHandlerCpu::VectorHandlerCpu()
  {
  }

  /**
   * @brief constructor
   *
   * @param new_workspace - workspace to be set
   */
  VectorHandlerCpu::VectorHandlerCpu(LinAlgWorkspaceCpu* new_workspace)
  {
    workspace_ = new_workspace;
  }

  /**
   * @brief destructor
   */
  VectorHandlerCpu::~VectorHandlerCpu()
  {
    // delete the workspace TODO
  }

  /**
   * @brief dot product of two vectors i.e, a = x^Ty
   *
   * @param[in] x The first vector
   * @param[in] y The second vector
   *
   * @return dot product (real number) of _x_ and _y_
   */

  real_type VectorHandlerCpu::dot(vector::Vector* x, vector::Vector* y)
  {
    const real_type* x_data = x->getData(memory::HOST);
    const real_type* y_data = y->getData(memory::HOST);
    real_type        sum    = 0.0;
    real_type        c      = 0.0;
    // real_type t, y;
    for (int i = 0; i < x->getSize(); ++i)
    {
      real_type y = (x_data[i] * y_data[i]) - c;
      real_type t = sum + y;
      c           = (t - sum) - y;
      sum         = t;
      // sum += (x_data[i] * y_data[i]);
    }
    return sum;
  }

  /**
   * @brief scale a vector by a constant i.e, x = alpha*x where alpha is a constant
   *
   * @param[in] alpha The constant
   * @param[in,out] x The vector
   *
   */
  void VectorHandlerCpu::scal(const real_type alpha, vector::Vector* x)
  {
    real_type* x_data = x->getData(memory::HOST);

    for (int i = 0; i < x->getSize(); ++i)
    {
      x_data[i] *= alpha;
    }
    x->setDataUpdated(memory::HOST);
  }

  /**
   * @brief compute infinity norm of a vector (i.e., find an entry with largest absolute value)
   *
   * @param[in] x vector
   *
   * @return infinity norm (real number) of _x_
   *
   */
  real_type VectorHandlerCpu::amax(vector::Vector* x)
  {
    const real_type* x_data = x->getData(memory::HOST);

    real_type vecmax = std::abs(x_data[0]);
    real_type v;
    for (int i = 1; i < x->getSize(); ++i)
    {
      v = std::abs(x_data[i]);
      if (v > vecmax)
      {
        vecmax = v;
      }
    }
    return vecmax;
  }

  /**
   * @brief axpy i.e, y = alpha*x+y where alpha is a constant
   *
   * @param[in] alpha The constant
   * @param[in] x The first vector
   * @param[in,out] y The second vector (result is return in y)
   *
   */
  void VectorHandlerCpu::axpy(const real_type alpha, /* const */ vector::Vector* x, vector::Vector* y)
  {
    // AXPY:  y = alpha * x + y
    real_type* x_data = x->getData(memory::HOST);
    real_type* y_data = y->getData(memory::HOST);
    for (int i = 0; i < x->getSize(); ++i)
    {
      y_data[i] = alpha * x_data[i] + y_data[i];
    }
    y->setDataUpdated(memory::HOST);
  }

  /**
   * @brief gemv computes matrix-vector product where both matrix and vectors are dense.
   *        i.e., x = beta*x +  alpha*V*y
   *
   * @param[in] Transpose - transposed = 'T' or not 'N'
   * @param[in] n Number of rows in (non-transposed) matrix
   * @param[in] k Number of columns in (non-transposed) matrix
   * @param[in] alpha Constant real number
   * @param[in] beta Constant real number
   * @param[in] V Multivector containing the matrix, organized columnwise
   * @param[in] y Vector, k x 1 if N and n x 1 if T
   * @param[in,out] x Vector, n x 1 if N and k x 1 if T
   *
   * @note Parameter k is not the total number of columns in V but the number
   * of columns to use in matrix-vector product.
   *
   * @pre _n_ > 0, _k_ > 0
   * @pre Number of columns in V >= k
   * @pre If transpose = N, size of y must equal k. If transpose = T, size of
   * x must equal k.
   *
   */
  void VectorHandlerCpu::gemv(char            transpose,
                              index_type      k,
                              const real_type alpha,
                              const real_type beta,
                              vector::Vector* V,
                              vector::Vector* y,
                              vector::Vector* x)
  {
    // x = beta*x +  alpha*V*y OR x = beta*x + alpha*V^Ty
    const real_type* V_data = V->getData(memory::HOST);
    const real_type* y_data = y->getData(memory::HOST);
    real_type*       x_data = x->getData(memory::HOST);
    const index_type n      = V->getSize();

    index_type i, j;
    real_type  sum;
    switch (transpose)
    {
    case 'T':
      assert((V->getSize() == y->getSize())
             && "gemv: Size mismatch! Size of V does not match size of y.");
      for (i = 0; i < k; ++i)
      {
        sum         = beta * x_data[i];
        real_type c = 0.0;
        for (j = 0; j < n; ++j)
        {
          real_type y = (alpha * V_data[i * n + j] * y_data[j]) - c;
          real_type t = sum + y;
          c           = (t - sum) - y;
          sum         = t;
          // sum += ((*alpha) * V_data[i * n + j] * y_data[j]);
        }
        x_data[i] = sum;
      }
      break;
    case 'N':
      assert((V->getSize() == x->getSize())
             && "gemv: Size mismatch! Size of V does not match size of x.");
      for (i = 0; i < n; ++i)
      {
        sum         = beta * x_data[i];
        real_type c = 0.0;
        for (j = 0; j < k; ++j)
        {
          real_type y = (alpha * V_data[n * j + i] * y_data[j]) - c;
          real_type t = sum + y;
          c           = (t - sum) - y;
          sum         = t;
          // sum += ((*alpha) * V_data[n * j + i] * y_data[j]);
        }
        x_data[i] = sum;
      }
      break;
    default:
      out::error() << "Unrecognized transpose option " << transpose
                   << " in gemv. Valid options are 'N' (not transposed) and 'T' (transposed).\n";
    } // switch
    x->setDataUpdated(memory::HOST);
    return;
  }

  /**
   * @brief gemm computes dense matrix-matrix (or multivector-multivector) product.
   *
   * Compute C := alpha * A * B + beta * C.
   * A is replaced with A^T if transpose_A = T.
   * B is replaced with B^T if transpose_A = T.
   *
   * @param[in] transpose_A - yes (T) or no (N)
   * @param[in] transpose_B - yes (T) or no (N)
   * @param[in] alpha     - Constant real number
   * @param[in] beta      - Constant real number
   * @param[in] A         - Multivector containing the A matrix, organized columnwise
   * @param[in] B         - Multivector containing the B matrix, organized columnwise
   * @param[in] C         - Multivector containing the C (result) matrix, organized columnwise
   */
  void VectorHandlerCpu::gemm(char transpose_A,
                              char transpose_B,
                              const real_type alpha,
                              const real_type beta,
                              vector::Vector* A,
                              vector::Vector* B,
                              vector::Vector* C)
  {
    const real_type* A_data = A->getData(memory::HOST);
    const real_type* B_data = B->getData(memory::HOST);
    real_type*       C_data = C->getData(memory::HOST);

    // Shape is post-transpose, if applicable
    index_type m = C->getSize();
    index_type n = C->getNumVectors();
    index_type k; // inner dimension

    switch (transpose_A)
    {
    case 'T':
      assert((A->getNumVectors() == m)
              && "gemm: Shape mismatch! Shape of A does not match shape of C.");
      k = A->getSize();
      break;
    case 'N':
      assert((A->getSize() == m)
              && "gemm: Shape mismatch! Shape of A does not match shape of C.");
      k = A->getNumVectors();
      break;
    default:
      out::error() << "Unrecognized transpose option " << transpose_A
                   << " in gemm. Valid options are 'N' (not transposed) and 'T' (transposed).\n";
      break;
    }

    switch (transpose_B)
    {
    case 'T':
      assert((B->getNumVectors() == k)
              && "gemm: Shape mismatch! Shape of A does not match shape of B.");
      assert((B->getSize() == n)
              && "gemm: Shape mismatch! Shape of A does not match shape of C.");
      break;
    case 'N':
      assert((B->getSize() == k)
              && "gemm: Shape mismatch! Shape of A does not match shape of B.");
      assert((B->getNumVectors() == n)
              && "gemm: Shape mismatch! Shape of A does not match shape of C.");
      break;
    default:
      out::error() << "Unrecognized transpose option " << transpose_B
                   << " in gemm. Valid options are 'N' (not transposed) and 'T' (transposed).\n";
      break;
    }

    index_type i, j, l;
    real_type  sum;
    switch (transpose_A)
    {
    case 'T':
      switch (transpose_B)
      {
      case 'T':
        for (i = 0; i < n; ++i)
        {
          for (j = 0; j < m; ++j)
          {
            sum         = beta * C_data[j + i * m];
            real_type c = 0.0;
            for (l = 0; l < k; ++l)
            {
              real_type y = (alpha * A_data[j * k + l] * B_data[i + l * n]) - c;
              real_type t = sum + y;
              c           = (t - sum) - y;
              sum         = t;
              // sum += ((*alpha) * A_data[j * k + l] * B_data[i + l * n]);
            }
            C_data[j + i * m] = sum;
          }
        }
        break;
      case 'N':
        for (i = 0; i < n; ++i)
        {
          for (j = 0; j < m; ++j)
          {
            sum         = beta * C_data[j + i * m];
            real_type c = 0.0;
            for (l = 0; l < k; ++l)
            {
              real_type y = (alpha * A_data[j * k + l] * B_data[l + i * k]) - c;
              real_type t = sum + y;
              c           = (t - sum) - y;
              sum         = t;
              // sum += ((*alpha) * A_data[j * k + l] * B_data[l + i * k]);
            }
            C_data[j + i * m] = sum;
          }
        }
        break;
      } // switch (transpose_B)
      break;
    
    case 'N':
      switch (transpose_B)
      {
      case 'T':
        for (i = 0; i < n; ++i)
        {
          for (j = 0; j < m; ++j)
          {
            sum         = beta * C_data[j + i * m];
            real_type c = 0.0;
            for (l = 0; l < k; ++l)
            {
              real_type y = (alpha * A_data[l * m + j] * B_data[i + l * n]) - c;
              real_type t = sum + y;
              c           = (t - sum) - y;
              sum         = t;
              // sum += ((*alpha) * A_data[l * m + j] * B_data[i + l * n]);
            }
            C_data[j + i * m] = sum;
          }
        }
        break;
      case 'N':
        for (i = 0; i < n; ++i)
        {
          for (j = 0; j < m; ++j)
          {
            sum         = beta * C_data[j + i * m];
            real_type c = 0.0;
            for (l = 0; l < k; ++l)
            {
              real_type y = (alpha * A_data[l * m + j] * B_data[l + i * k]) - c;
              real_type t = sum + y;
              c           = (t - sum) - y;
              sum         = t;
              // sum += ((*alpha) * A_data[l * m + j] * B_data[l + i * k]);
            }
            C_data[j + i * m] = sum;
          }
        }
        break;
      } // switch (transpose_B)
      break;
    } // switch (transpose_A)

    C->setDataUpdated(memory::HOST);
    return;
  }

  /**
   * @brief mass (bulk) axpy i.e, y = y - x*alpha where  alpha is a vector
   *
   * @param[in] size number of elements in y
   * @param[in] alpha vector size k x 1
   * @param[in] x (multi)vector size size x k
   * @param[in,out] y vector size size x 1 (this is where the result is stored)
   *
   * @pre   _k_ > 0, _size_ > 0, _size_ = x->getSize()
   *
   */
  void VectorHandlerCpu::axpyMulti(index_type      size,
                                   vector::Vector* alpha,
                                   index_type      k,
                                   vector::Vector* x,
                                   vector::Vector* y)
  {

    real_type* alpha_data = alpha->getData(memory::HOST);
    real_type* y_data     = y->getData(memory::HOST);
    real_type* x_data     = x->getData(memory::HOST);
    index_type i, j;
    real_type  sum;

    for (i = 0; i < size; ++i)
    {
      sum = 0.0;
      for (j = 0; j < k; ++j)
      {
        sum += x_data[j * size + i] * alpha_data[j];
      }
      y_data[i] = y_data[i] - sum;
    }
    y->setDataUpdated(memory::HOST);
  }

  /**
   * @brief mass (bulk) dot product i.e,  V^T x, where V is n x k dense multivector (a dense multivector consisting of k vectors size n)
   *        and x is k x 2 dense multivector (a multivector consisiting of two vectors size n each)
   *
   * @param[in] size Number of elements in a single vector in V
   * @param[in] V Multivector; k vectors size n x 1 each
   * @param[in] k Number of vectors in V
   * @param[in] x Multivector; 2 vectors size n x 1 each
   * @param[out] res Multivector; 2 vectors size k x 1 each (result is returned in res)
   *
   * @pre   _size_ > 0, _k_ > 0, size = x->getSize(), _res_ needs to be allocated
   *
   */
  void VectorHandlerCpu::dot2Multi(index_type      size,
                                   vector::Vector* V,
                                   index_type      q,
                                   vector::Vector* x,
                                   vector::Vector* res)
  {
    real_type*       res_data = res->getData(memory::HOST);
    const real_type* x_data   = x->getData(memory::HOST);
    const real_type* V_data   = V->getData(memory::HOST);

    real_type c0 = 0.0;
    real_type cq = 0.0;

    for (index_type i = 0; i < q; ++i)
    {
      res_data[i]     = 0.0;
      res_data[i + q] = 0.0;

      // Make sure we don't accumulate round-off errors
      for (index_type j = 0; j < size; ++j)
      {
        real_type y0 = (V_data[i * size + j] * x_data[j]) - c0;
        real_type yq = (V_data[i * size + j] * x_data[j + size]) - cq;
        real_type t0 = res_data[i] + y0;
        real_type tq = res_data[i + q] + yq;
        c0           = (t0 - res_data[i]) - y0;
        cq           = (tq - res_data[i + q]) - yq;

        res_data[i]     = t0;
        res_data[i + q] = tq;
      }
    }
    res->setDataUpdated(memory::HOST);
  }

  /**
   * @brief Scale a vector by a diagonal matrix
   *
   * @param[in] diag Diagonal vector
   * @param[in] vec Vector to be scaled
   *
   * @return 0 if successful, 1 otherwise
   */
  void VectorHandlerCpu::scal(vector::Vector* diag, vector::Vector* vec)
  {
    const real_type* diag_data = diag->getData(memory::HOST);
    real_type*       vec_data  = vec->getData(memory::HOST);
    index_type       n         = vec->getSize();

    for (index_type i = 0; i < n; ++i)
    {
      vec_data[i] *= diag_data[i];
    }
    vec->setDataUpdated(memory::HOST);
  }

  /**
   * @brief Scale a vector by a diagonal matrix
   *
   * @param[in] diag Diagonal vector
   * @param[in] vec Vector to be scaled
   * @param[in] diag_offset - the index of diag where the diagonal matrix begins
   *
   * @return 0 if successful, 1 otherwise
   */
  void VectorHandlerCpu::scal(vector::Vector* diag, vector::Vector* vec, index_type diag_offset)
  {
    const real_type* diag_data = &diag->getData(memory::HOST)[diag_offset];
    real_type*       vec_data  = vec->getData(memory::HOST);
    index_type       n         = vec->getSize();

    for (index_type i = 0; i < n; ++i)
    {
      vec_data[i] *= diag_data[i];
    }
    vec->setDataUpdated(memory::HOST);
  }

  // temporary
  index_type idxmap(index_type i, index_type j, index_type col_length)
  {
    return i * (col_length) + j;
  }

  // // ... grabbed this off the internet. not production code
  // int VectorHandlerCpu::choleskyFactorize(const vector::Vector* A, real_type* out)
  // {
  //   int n = A->getSize();
  //   const real_type* A_data = A->getData(memory::HOST);

  //   // Decomposing a matrix into Lower Triangular
  //   for (int i = 0; i < n; i++) {
  //       for (int j = 0; j <= i; j++) {
  //           int sum = 0;

  //           // summation for diagonals
  //           if (j == i) {
  //               for (int k = 0; k < j; k++)
  //                   sum += pow(out[idxmap(j, k, n)], 2);
  //               out[idxmap(j, j, n)] = sqrt(A_data[idxmap(j, j, n)] - sum);
  //           } else {

  //               // Evaluating L(i, j) using L(j, j)
  //               for (int k = 0; k < j; k++)
  //                   sum += (out[idxmap(i, k, n)] * out[idxmap(j, k, n)]);
  //               out[idxmap(i, j, n)] = (A_data[idxmap(i, j, n)] - sum) / out[idxmap(j, j, n)];
  //           }
  //       }
  //   }
  //   return 0;
  // }

  /**
   * @brief Multiplies vector by an inverse of a diagonal matrix.
   *
   * @param[in]  diag   - diagonal matrix stored in a vector object
   * @param[in,out] vec - vector to be divided
   *
   * @pre The two vectors must be the same size
   *
   * @return 0 if successful, 1 otherwise
   */
  int VectorHandlerCpu::diagSolve(vector::Vector* diag, vector::Vector* vec)
  {
    real_type* diag_data = diag->getData(memory::HOST);
    real_type* vec_data  = vec->getData(memory::HOST);
    index_type n         = vec->getSize();

    for (index_type i = 0; i < n; ++i)
    {
      vec_data[i] /= diag_data[i];
    }
    vec->setDataUpdated(memory::HOST);
    return 0;
  }

  /**
   * @brief Take the element-wise max of two vectors.
   * Each element of the output will be the maximum value of the corresponding elements in the input vectors.
   *
   * @param[in]  x   - First input vector
   * @param[in]  y   - Second input vector
   * @param[out] out - Output vector
   *
   * @pre The three vectors must be the same size
   *
   * @return 0 if successful, 1 otherwise
   */
  int VectorHandlerCpu::max(/* const */ vector::Vector* x, /* const */ vector::Vector* y, vector::Vector* out)
  {
    const real_type* x_data   = x->getData(memory::HOST);
    const real_type* y_data   = y->getData(memory::HOST);
    real_type*       out_data = out->getData(memory::HOST);
    index_type       n        = y->getSize();

    for (index_type i = 0; i < n; ++i)
    {
      out_data[i] = std::max(x_data[i], y_data[i]);
    }
    out->setDataUpdated(memory::HOST);
    return 0;
  }

  /**
   * @brief Take the element-wise absolute value of a vector.
   *
   * @param[in,out] x - Input and output vector
   *
   * @return 0 if successful, 1 otherwise
   */
  int VectorHandlerCpu::abs(/* const */ vector::Vector* in, vector::Vector* out)
  {
    const real_type* in_data  = in->getData(memory::HOST);
    real_type*       out_data = out->getData(memory::HOST);
    index_type       n        = in->getSize();

    for (index_type i = 0; i < n; ++i)
    {
      out_data[i] = std::abs(in_data[i]);
    }
    out->setDataUpdated(memory::HOST);
    return 0;
  }

} // namespace ReSolve
