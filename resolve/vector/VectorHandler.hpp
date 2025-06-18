#pragma once
#include <resolve/MemoryUtils.hpp>

namespace ReSolve
{
  namespace vector
  {
    class Vector;
  }
  class VectorHandlerImpl;
  class LinAlgWorkspaceCpu;
  class LinAlgWorkspaceCUDA;
  class LinAlgWorkspaceHIP;
} // namespace ReSolve

namespace ReSolve
{ // namespace vector {

  class VectorHandler
  {
  public:
    VectorHandler();
    VectorHandler(LinAlgWorkspaceCpu* new_workspace);
    VectorHandler(LinAlgWorkspaceCUDA* new_workspace);
    VectorHandler(LinAlgWorkspaceHIP* new_workspace);
    ~VectorHandler();

    // y = alpha x + y
    void axpy(const real_type* alpha, vector::Vector* x, vector::Vector* y, memory::MemorySpace memspace);

    // dot: x \cdot y
    real_type dot(vector::Vector* x, vector::Vector* y, memory::MemorySpace memspace);

    // scal = alpha * x
    void scal(const real_type* alpha, vector::Vector* x, memory::MemorySpace memspace);

    // mass axpy: x*alpha + y where x is [n x k] and alpha is [k x 1]; x is stored columnwise
    void massAxpy(index_type size, vector::Vector* alpha, index_type k, vector::Vector* x, vector::Vector* y, memory::MemorySpace memspace);

    // mass dot: V^T x, where V is [n x k] and x is [k x 2], everything is stored and returned columnwise
    // Size = n
    void massDot2Vec(index_type size, vector::Vector* V, index_type k, vector::Vector* x, vector::Vector* res, memory::MemorySpace memspace);

    /** gemv:
     * if `transpose = N` (no), `x = beta*x +  alpha*V*y`,
     * where `x` is `[n x 1]`, `V` is `[n x k]` and `y` is `[k x 1]`.
     * if `transpose = T` (yes), `x = beta*x + alpha*V^T*y`,
     * where `x` is `[k x 1]`, `V` is `[n x k]` and `y` is `[n x 1]`.
     */
    void gemv(char                transpose,
              index_type          n,
              index_type          k,
              const real_type*    alpha,
              const real_type*    beta,
              vector::Vector*     V,
              vector::Vector*     y,
              vector::Vector*     x,
              memory::MemorySpace memspace);

    int scale(vector::Vector* diag, vector::Vector* vec, memory::MemorySpace memspace);

    /** infNorm:
     * Returns infinity norm of a vector (i.e., entry with max abs value)
     */
    real_type infNorm(vector::Vector* x, memory::MemorySpace memspace);

    bool getIsCudaEnabled() const;
    bool getIsHipEnabled() const;

  private:
    VectorHandlerImpl* cpuImpl_{nullptr};
    VectorHandlerImpl* devImpl_{nullptr}; ///< Pointer to device implementation

    bool isCpuEnabled_{false};
    bool isCudaEnabled_{false};
    bool isHipEnabled_{false};
  };

} // namespace ReSolve
