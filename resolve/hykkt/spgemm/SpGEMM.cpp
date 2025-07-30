#include "SpGEMM.hpp"

#include "SpGEMMCpu.hpp"
#ifdef RESOLVE_USE_CUDA
#include "SpGEMMCuda.hpp"
#elif defined(RESOLVE_USE_HIP)
#include "SpGEMMHip.hpp"
#endif

namespace ReSolve {
  namespace hykkt {
    SpGEMM::SpGEMM(memory::MemorySpace memspace): memspace_(memspace)
    {
      if (memspace_ == memory::HOST)
      {
        impl_ = new SPGEMMCpu();
      }
      else
      {
#ifdef RESOLVE_USE_CUDA
        impl_ = new SPGEMMCuda();
#elif defined(RESOLVE_USE_HIP)
        impl_ = new SPGEMMHip();
#else
        out::error() << "No GPU support enabled, and memory space set to DEVICE.\n";
        exit(1);
#endif
      }
    }

    SpGEMM::~SpGEMM() {
      delete impl_;
    }

    void SpGEMM::addProductMatrices(matrix::Csr* A, matrix::Csr* B) {
      impl_->addProductMatrices(A, B);
    }

    void SpGEMM::addSumMatrix(matrix::Csr* D) {
      impl_->addSumMatrix(D);
    }

    void SpGEMM::addResultMatrix(matrix::Csr* E) {
      impl_->addResultMatrix(E);
    }

    void SpGEMM::compute() {
      impl_->compute();
    }
  }
}