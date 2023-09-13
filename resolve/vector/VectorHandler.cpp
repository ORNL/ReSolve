#include <iostream>
#include <resolve/cudaKernels.h>
#include <resolve/vector/Vector.hpp>

#include "VectorHandler.hpp"

namespace ReSolve {

  /* 
   * @brief empty constructor that does absolutely nothing        
   */
  VectorHandler::VectorHandler()
  {
  }

  /* 
   * @brief constructor
   * 
   * @param new_workspace - workspace to be set     
   */
  VectorHandler:: VectorHandler(LinAlgWorkspace* new_workspace)
  {
    workspace_ = new_workspace;
  }

  /* 
   * @brief destructor     
   */
  VectorHandler::~VectorHandler()
  {
    //delete the workspace TODO
  }

  /* 
   * @brief dot product of two vectors i.e, a = x^Ty
   * 
   * @param x - the first vector
   *        y - the second vector
   *        memspace - string containg memspace (cpu or cuda)
   * 
   * @return dot product (real number)
   */
  real_type VectorHandler::dot(vector::Vector* x, vector::Vector* y, std::string memspace)
  { 
    if (memspace == "cuda" ){ 
      LinAlgWorkspaceCUDA* workspaceCUDA = (LinAlgWorkspaceCUDA*) workspace_;
      cublasHandle_t handle_cublas =  workspaceCUDA->getCublasHandle();
      double nrm = 0.0;
      cublasStatus_t st= cublasDdot (handle_cublas,  x->getSize(), x->getData("cuda"), 1, y->getData("cuda"), 1, &nrm);
      if (st!=0) {printf("dot product crashed with code %d \n", st);}
      return nrm;
    } else {
      if (memspace == "cpu") {
        real_type* x_data = x->getData("cpu");
        real_type* y_data = y->getData("cpu");
        real_type sum = 0.0;
        real_type c;
        real_type t, y;
        for (int i = 0; i < x->getSize(); ++i){
          y = (x_data[i] * y_data[i]) - c;
          t = sum + y;
          c = (t - sum) - y;
          sum = t;        
          //   sum += (x_data[i] * y_data[i]);
        } 
        return sum;
      } else {
        std::cout<<"Not implemented (yet)"<<std::endl;
        return NAN;
      }
    }
  }

  /* 
   * @brief scale a vector by a constant i.e, x = alpha*x where alpha is a constant
   * 
   * @param alpha - the constant
   *        x - the vector
   *        memspace - string containg memspace (cpu or cuda)
   * 
   */
  void VectorHandler::scal(const real_type* alpha, vector::Vector* x, std::string memspace)
  {
    if (memspace == "cuda" ) { 
      LinAlgWorkspaceCUDA* workspaceCUDA = (LinAlgWorkspaceCUDA*) workspace_;
      cublasHandle_t handle_cublas =  workspaceCUDA->getCublasHandle();
      cublasStatus_t st = cublasDscal(handle_cublas, x->getSize(), alpha, x->getData("cuda"), 1);
      if (st!=0) {printf("scal crashed with code %d \n", st);}

    } else {
      if (memspace == "cpu") {
        real_type* x_data = x->getData("cpu");

        for (int i = 0; i < x->getSize(); ++i){
          x_data[i] *= (*alpha);
        }
      } else {      
        std::cout<<"Not implemented (yet)"<<std::endl;
      }  
    }
  }

  /* 
   * @brief axpy i.e, y = alpha*x+y where alpha is a constant
   * 
   * @param alpha - the constant
   *        x - the first vector
   *        y - the second vector (result is return in y)
   *        memspace - string containg memspace (cpu or cuda)
   * 
   */
  void VectorHandler::axpy(const  real_type* alpha, vector::Vector* x, vector::Vector* y, std::string memspace )
  {
    //AXPY:  y = alpha * x + y
    if (memspace == "cuda" ) { 
      LinAlgWorkspaceCUDA* workspaceCUDA = (LinAlgWorkspaceCUDA*) workspace_;
      cublasHandle_t handle_cublas =  workspaceCUDA->getCublasHandle();
      cublasDaxpy(handle_cublas,
                  x->getSize(),
                  alpha,
                  x->getData("cuda"),
                  1,
                  y->getData("cuda"),
                  1);
    } else {
      if (memspace == "cpu") {
        real_type* x_data = x->getData("cpu");
        real_type* y_data = y->getData("cpu");
        for (int i = 0; i < x->getSize(); ++i){
          y_data[i] = (*alpha) * x_data[i] + y_data[i];
        }
      } else {
        std::cout << "Not implemented (yet)" << std::endl;

      }
    }
  }

  /* 
   * @brief gemv computes matrix-vector product where both matrix and vectors are dense.
   *        i.e., x = beta*x +  alpha*V*y
   *
   * @param transpose - yes (T) or no (N)
   *        n - number of rows in (non-transposed) matrix
   *        k - number of columns in (non-transposed)   
   *        alpha - constant real number
   *        beta - constant real number
   *        V - multivector containing the matrix, organized columnwise
   *        y - vector, k x 1 if N and n x 1 if T
   *        x - vector, n x 1 if N and k x 1 if T
   *        memspace - cpu or cuda (for now)
   *
   * @pre   V is stored colum-wise, n, k > 0
   * 
   */  
  void VectorHandler::gemv(std::string transpose, index_type n, index_type k, const real_type* alpha, const real_type* beta, vector::Vector* V, vector::Vector* y, vector::Vector* x, std::string memspace)
  {
    if (memspace == "cuda") {
      LinAlgWorkspaceCUDA* workspaceCUDA = (LinAlgWorkspaceCUDA*) workspace_;
      cublasHandle_t handle_cublas =  workspaceCUDA->getCublasHandle();
      if (transpose == "T") {

        cublasDgemv(handle_cublas,
                    CUBLAS_OP_T,
                    n,
                    k,
                    alpha,
                    V->getData("cuda"),
                    n,
                    y->getData("cuda"),
                    1,
                    beta,
                    x->getData("cuda"),
                    1);

      } else {
        cublasDgemv(handle_cublas,
                    CUBLAS_OP_N,
                    n,
                    k,
                    alpha,
                    V->getData("cuda"),
                    n,
                    y->getData("cuda"),
                    1,
                    beta,
                    x->getData("cuda"),
                    1);
      }

    } else {

      std::cout<<"Not implemented (yet)"<<std::endl;
    }
  }

  /* 
   * @brief mass (bulk) axpy i.e, y = alpha*x where  alpha is a vector
   * 
   * @param size - number of elements in y
   *        alpha - vector size k x 1
   *        x - vector size size x 1
   *        y - vector size size x 1 (this is where the result is stored)
   *        memspace - string containg memspace (cpu or cuda)
   * @pre   k, size > 0, size = x->getSize()
   *
   */
  void VectorHandler::massAxpy(index_type size, vector::Vector* alpha, index_type k, vector::Vector* x, vector::Vector* y, std::string memspace)
  {
    if (memspace == "cuda") {
      if (k < 200) {
        mass_axpy(size, k, x->getData("cuda"), y->getData("cuda"),alpha->getData("cuda"));
      } else {
        LinAlgWorkspaceCUDA* workspaceCUDA = (LinAlgWorkspaceCUDA*) workspace_;
        cublasHandle_t handle_cublas =  workspaceCUDA->getCublasHandle();
        cublasDgemm(handle_cublas,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    size,       // m
                    1,          // n
                    k + 1,      // k
                    &minusone, // alpha
                    x->getData("cuda"), // A
                    size,       // lda
                    alpha->getData("cuda"), // B
                    k + 1,      // ldb
                    &one,
                    y->getData("cuda"),          // c
                    size);      // ldc     
      }
    } else {
      std::cout<<"Not implemented (yet)"<<std::endl;
    }
  }

  //mass dot: everything is stored and returned columnwise
  /* 
   * @brief mass (bulk) dot product i.e,  V^T x, where V is n x k dense multivector (a dense multivector consisting of k vectors size n)  
   *        and x is k x 2 dense multivector (a multivector consisiting of two vectors size n each)
   * 
   * @param size - number of elements in a single vector in V
   *        V - multivector; k vectors size n x 1 each
   *        k - number of vectors in V
   *        x - multivector; 2 vectors size n x 1 each
   *        res - multivector; 2 vectors size k x 1 each (result is returned in res)
   *        memspace - string containg memspace (cpu or cuda)
   *
   * @pre   sizei, k > 0, size = x->getSize()
   *
   */
  void VectorHandler::massDot2Vec(index_type size, vector::Vector* V, index_type k, vector::Vector* x, vector::Vector* res, std::string memspace)
  {

    if (memspace == "cuda") {
      if (k < 200) {
        mass_inner_product_two_vectors(size, k, x->getData("cuda") , x->getData(1, "cuda"), V->getData("cuda"), res->getData("cuda"));
      } else {
        LinAlgWorkspaceCUDA* workspaceCUDA = (LinAlgWorkspaceCUDA*) workspace_;
        cublasHandle_t handle_cublas =  workspaceCUDA->getCublasHandle();
        cublasDgemm(handle_cublas,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    k + 1,   //m
                    2,       //n
                    size,    //k
                    &one,   //alpha
                    V->getData("cuda"),       //A
                    size,    //lda
                    x->getData("cuda"),       //B
                    size,    //ldb
                    &zero,
                    res->getData("cuda"),     //c
                    k + 1);  //ldc 
      }
    } else {
      std::cout<<"Not implemented (yet)"<<std::endl;
    }
  }

} // namespace ReSolve
