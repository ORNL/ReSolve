#pragma once

#include <resolve/Common.hpp>
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
{
  class VectorHandler
  {
  public:
    VectorHandler();
    VectorHandler(LinAlgWorkspaceCpu* new_workspace);
    VectorHandler(LinAlgWorkspaceCUDA* new_workspace);
    VectorHandler(LinAlgWorkspaceHIP* new_workspace);
    ~VectorHandler();

    // y := alpha x + y
    void axpy(const real_type             alpha,
              /* const */ vector::Vector* x,
              vector::Vector*             y,
              memory::MemorySpace         memspace);

    // Dot product of two vectors
    real_type dot(vector::Vector* x, vector::Vector* y, memory::MemorySpace memspace);

    // Scale vector by scalar
    void scal(const real_type alpha, vector::Vector* x, memory::MemorySpace memspace);

    // Scale vector by diagonal matrix represented as a vector (i.e., vec = diag*vec)
    void scal(vector::Vector* diag, vector::Vector* vec, memory::MemorySpace memspace);
    void scal(vector::Vector*     diag,
              vector::Vector*     vec,
              index_type          diag_offset,
              memory::MemorySpace memspace);

    // axpy for multivectors
    void axpyMulti(index_type          size,
                   vector::Vector*     alpha,
                   index_type          k,
                   vector::Vector*     x,
                   vector::Vector*     y,
                   memory::MemorySpace memspace);

    // Dot product of two vectors with a multivector V
    void dot2Multi(index_type          size,
                   vector::Vector*     V,
                   index_type          k,
                   vector::Vector*     x,
                   vector::Vector*     res,
                   memory::MemorySpace memspace);

    // Dense matrix-vector product.
    void gemv(char                transpose,
              index_type          k, // number of vectors from multivector V to use
              const real_type     alpha,
              const real_type     beta,
              vector::Vector*     V,
              vector::Vector*     y,
              vector::Vector*     x,
              memory::MemorySpace memspace);

    int diagSolve(vector::Vector* diag, vector::Vector* vec, memory::MemorySpace memspace);
    int max(/* const */ vector::Vector* x, /* const */ vector::Vector* y, vector::Vector* out, memory::MemorySpace memspace);

    int abs(/* const */ vector::Vector* in, vector::Vector* out, memory::MemorySpace memspace);

    // Vector infinity norm
    real_type amax(vector::Vector* x, memory::MemorySpace memspace);

    bool getIsCudaEnabled() const;
    bool getIsHipEnabled() const;

  private:
    VectorHandlerImpl* cpuImpl_{nullptr};
    VectorHandlerImpl* devImpl_{nullptr}; ///< Pointer to device implementation

    bool isCpuEnabled_{false};
    bool isCudaEnabled_{false};
    bool isHipEnabled_{false};
  }; // class VectorHandler

} // namespace ReSolve
