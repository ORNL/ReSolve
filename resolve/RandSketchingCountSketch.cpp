#include "RandSketchingCountSketch.hpp"
#include <resolve/MemoryUtils.hpp>
#include <resolve/vector/Vector.hpp>

#ifdef RESOLVE_USE_HIP
#include <resolve/hip/hipKernels.h>
#elif RESOLVE_USE_CUDA
#include <resolve/cuda/cudaKernels.h>
#else
#include <resolve/cpu/cpuKernels.h>
#endif
#include <resolve/RandSketchingCountSketch.hpp> 

namespace ReSolve 
{
  RandSketchingCountSketch::RandSketchingCountSketch(memory::MemorySpace memspace)
    : memspace_(memspace)
  {
    h_labels_ = nullptr;
    h_flip_ = nullptr;

// #if defined(RESOLVE_USE_CUDA) || defined(RESOLVE_USE_HIP) 
    d_labels_ = nullptr;
    d_flip_ = nullptr;
// #endif
  }

  // destructor
  RandSketchingCountSketch::~RandSketchingCountSketch()
  {
    delete h_labels_;
    delete h_flip_;
// #if defined(RESOLVE_USE_CUDA) || defined(RESOLVE_USE_HIP)
    if (memspace_ == memory::DEVICE) {
      mem_.deleteOnDevice(d_labels_);
      mem_.deleteOnDevice(d_flip_);
    }
// #endif
  }

  // Actual sketching process
  int RandSketchingCountSketch::Theta(vector_type* input, vector_type* output)
  {
    using namespace memory;
   
// #if defined(RESOLVE_USE_CUDA) || defined(RESOLVE_USE_HIP)
    switch (memspace_) {
      case DEVICE:
        mem_.deviceSynchronize();
        count_sketch_theta(n_,
                          k_rand_,
                          d_labels_,
                          d_flip_,
                          input->getData(ReSolve::memory::DEVICE), 
                          output->getData(ReSolve::memory::DEVICE));
        mem_.deviceSynchronize();
        break;
      case HOST:
// #else // cpu only
        count_sketch_theta(n_,
                          k_rand_,
                          h_labels_,
                          h_flip_,
                          input->getData(ReSolve::memory::HOST), 
                          output->getData(ReSolve::memory::HOST));
        break;
    }
// #endif

    return 0;
  }

  // Setup the parameters, sampling matrices, permuations, etc
  int RandSketchingCountSketch::setup(index_type n, index_type k)
  {
    k_rand_ = k;
    n_ = n;
    srand(static_cast<unsigned>(time(nullptr)));
    //allocate labeling scheme vector and move to GPU

    h_labels_ = new int[n_];
    //allocate sgn - a vector of flip signs
    h_flip_  = new int[n_];

    //populate labeling scheme (can be done on the gpu really)
    //to be fixed, this can be done on the GPU
    for (int i=0; i<n; ++i) {
      h_labels_[i] = rand() % k_rand_;
      int r = rand()%100;
      if (r < 50){
        h_flip_[i] = -1;
      } else { 
        h_flip_[i] = 1;
      }
    }

    using namespace memory;
   
    switch (memspace_) {
      case DEVICE:
        mem_.allocateArrayOnDevice(&d_labels_, n_); 
        mem_.allocateArrayOnDevice(&d_flip_, n_); 
        //then copy
        mem_.copyArrayHostToDevice(d_labels_, h_labels_, n_);
        mem_.copyArrayHostToDevice(d_flip_, h_flip_, n_);
        mem_.deviceSynchronize();
        break;
      case HOST:
        break;
    }
// #if defined(RESOLVE_USE_CUDA) || defined(RESOLVE_USE_HIP) 
//     mem_.allocateArrayOnDevice(&d_labels_, n_); 
//     mem_.allocateArrayOnDevice(&d_flip_, n_); 

//     //then copy

//     mem_.copyArrayHostToDevice(d_labels_, h_labels_, n_);
//     mem_.copyArrayHostToDevice(d_flip_, h_flip_, n_);

//     mem_.deviceSynchronize();
// #endif
    return 0;
  }

  //to be fixed, this can be done on the GPU
  int RandSketchingCountSketch::reset() // if needed can be reset (like when Krylov method restarts)
  {
    for (int i = 0; i < n_; ++i) {
      h_labels_[i] = rand() % k_rand_;

      int r = rand()%100;
      if (r < 50){
        h_flip_[i] = -1;
      } else { 
        h_flip_[i] = 1;
      }
    }
// #if defined(RESOLVE_USE_CUDA) || defined(RESOLVE_USE_HIP)
    if (memspace_ == memory::DEVICE) {
      mem_.copyArrayHostToDevice(d_labels_, h_labels_, n_);
      mem_.copyArrayHostToDevice(d_flip_, h_flip_, n_);

      mem_.deviceSynchronize();
    }
// #endif
    return 0;
  }
}
