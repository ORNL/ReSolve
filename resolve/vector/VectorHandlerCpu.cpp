#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/cuda/cudaKernels.h>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/vector/VectorHandlerImpl.hpp>
#include "VectorHandlerCpu.hpp"

namespace ReSolve {
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
  VectorHandlerCpu:: VectorHandlerCpu(LinAlgWorkspaceCpu* new_workspace)
  {
    workspace_ = new_workspace;
  }

  /** 
   * @brief destructor     
   */
  VectorHandlerCpu::~VectorHandlerCpu()
  {
    //delete the workspace TODO
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
    real_type* x_data = x->getData(memory::HOST);
    real_type* y_data = y->getData(memory::HOST);
    real_type sum = 0.0;
    real_type c = 0.0;
    // real_type t, y;
    for (int i = 0; i < x->getSize(); ++i) {
      real_type y = (x_data[i] * y_data[i]) - c;
      real_type t = sum + y;
      c = (t - sum) - y;
      sum = t;        
      //sum += (x_data[i] * y_data[i]);
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
  void VectorHandlerCpu::scal(const real_type* alpha, vector::Vector* x)
  {
    real_type* x_data = x->getData(memory::HOST);

    for (int i = 0; i < x->getSize(); ++i){
      x_data[i] *= (*alpha);
    }
  }
  
  /** 
   * @brief compute infinity norm of a vector (i.e., find an entry with largest absolute value)
   * 
   * @param[in] The vector
   *
   * @return infinity norm (real number) of _x_
   * 
   */
  real_type VectorHandlerCpu::infNorm(vector::Vector* x)
  {
    real_type* x_data = x->getData(memory::HOST);
    real_type  vecmax = std::abs(x_data[0]);
    real_type v;
    for (int i = 1; i < x->getSize(); ++i){
     v = std::abs(x_data[i]);
     if (v > vecmax){
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
  void VectorHandlerCpu::axpy(const  real_type* alpha, vector::Vector* x, vector::Vector* y)
  {
    //AXPY:  y = alpha * x + y
    real_type* x_data = x->getData(memory::HOST);
    real_type* y_data = y->getData(memory::HOST);
    for (int i = 0; i < x->getSize(); ++i) {
      y_data[i] = (*alpha) * x_data[i] + y_data[i];
    }
  }

  /** 
   * @brief gemv computes matrix-vector product where both matrix and vectors are dense.
   *        i.e., x = beta*x +  alpha*V*y
   *
   * @param[in] Transpose - transposed = 'T' or not 'N'
   * @param[in] n Number of rows in (non-transposed) matrix
   * @param[in] k Number of columns in (non-transposed)   
   * @param[in] alpha Constant real number
   * @param[in] beta Constant real number
   * @param[in] V Multivector containing the matrix, organized columnwise
   * @param[in] y Vector, k x 1 if N and n x 1 if T
   * @param[in,out] x Vector, n x 1 if N and k x 1 if T
   *
   * @pre   V is stored colum-wise, _n_ > 0, _k_ > 0
   * 
   */  
  void VectorHandlerCpu::gemv(char transpose,
                              index_type n,
                              index_type k,
                              const real_type* alpha,
                              const real_type* beta,
                              vector::Vector* V,
                              vector::Vector* y,
                              vector::Vector* x)
  {
    // x = beta*x +  alpha*V*y OR x = beta*x + alpha*V^Ty
    real_type* V_data = V->getData(memory::HOST);
    real_type* y_data = y->getData(memory::HOST);
    real_type* x_data = x->getData(memory::HOST);
    index_type i, j;
    real_type sum;
    switch (transpose) {
      case 'T':
        for (i = 0; i < k; ++i) {
          sum = (*beta) * x_data[i]; 
          real_type c = 0.0;
          for (j = 0; j < n; ++j) {
            real_type y = ((*alpha) * V_data[i * n + j] * y_data[j]) - c;
            real_type t = sum + y;
            c = (t - sum) - y;
            sum = t;        
            //sum += ((*alpha) * V_data[i * n + j] * y_data[j]); 
          }
          x_data[i] = sum;
        }
        break;
      default:
        for (i = 0; i < n; ++i) {
          sum = (*beta) * x_data[i] ; 
          real_type c = 0.0;
          for (j = 0; j < k; ++j) {
            real_type y = ((*alpha) * V_data[n * j + i] * y_data[j]) - c;
            real_type t = sum + y;
            c = (t - sum) - y;
            sum = t;        
            //sum += ((*alpha) * V_data[n * j + i] * y_data[j]); 
          }
          x_data[i] = sum;
        }
        break;
        if (transpose != 'N') {
          out::warning() << "Unrecognized transpose option " << transpose
                         << " in gemv. Using non-transposed multivector.\n";
        }
    } // switch
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
  void VectorHandlerCpu::massAxpy(index_type size, 
                                  vector::Vector* alpha, 
                                  index_type k, 
                                  vector::Vector* x, 
                                  vector::Vector* y)
  {
  
    real_type* alpha_data = alpha->getData(memory::HOST);
    real_type* y_data = y->getData(memory::HOST);
    real_type* x_data = x->getData(memory::HOST);
    index_type i, j;
    real_type sum;
    
    for (i = 0; i < size; ++i) {
      sum = 0.0;
      for (j = 0; j < k; ++j) {
        sum += x_data[j * size + i] * alpha_data[j];
      }
      y_data[i] = y_data[i] - sum;
    }
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
  void VectorHandlerCpu::massDot2Vec(index_type size, 
                                     vector::Vector* V, 
                                     index_type q, 
                                     vector::Vector* x, 
                                     vector::Vector* res)
  {
    real_type* res_data = res->getData(memory::HOST);
    real_type* x_data   = x->getData(memory::HOST);
    real_type* V_data   = V->getData(memory::HOST);

    real_type c0 = 0.0;
    real_type cq = 0.0;

    for (index_type i = 0; i < q; ++i) {
      res_data[i] = 0.0; 
      res_data[i + q] = 0.0;

      // Make sure we don't accumulate round-off errors
      for (index_type j = 0; j < size; ++j) {
        real_type y0 = (V_data[i * size + j] * x_data[j])        - c0;    
        real_type yq = (V_data[i * size + j] * x_data[j + size]) - cq;
        real_type t0 = res_data[i]     + y0;
        real_type tq = res_data[i + q] + yq;
        c0 = (t0 - res_data[i]    ) - y0;
        cq = (tq - res_data[i + q]) - yq;

        res_data[i]     = t0;
        res_data[i + q] = tq;
      }
    }
  }

} // namespace ReSolve
