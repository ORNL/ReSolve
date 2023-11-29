#include "RandSketchingCountSketch.hpp"
#include <resolve/MemoryUtils.hpp>
#include <resolve/vector/Vector.hpp>

#ifdef RESOLVE_USE_HIP
#include <resolve/hip/hipKernels.h>
#endif
#ifdef RESOLVE_USE_CUDA
#include <resolve/cuda/cudaKernels.h>
#endif
#include <resolve/RandSketchingCountSketch.hpp> 

namespace ReSolve 
{
  RandSketchingCountSketch::RandSketchingCountSketch()
  {
    h_labels_ = nullptr;
    h_flip_ = nullptr;

    d_labels_ = nullptr;
    d_flip_ = nullptr;
  }

  // destructor
  RandSketchingCountSketch::~RandSketchingCountSketch()
  {
    delete h_labels_;
    delete h_flip_;
    mem_.deleteOnDevice(d_labels_);
    mem_.deleteOnDevice(d_flip_);
  }

  // Actual sketching process
  int RandSketchingCountSketch::Theta(vector_type* input, vector_type* output)
  {
    mem_.deviceSynchronize();
    count_sketch_theta(n_,
                       k_rand_,
                       d_labels_,
                       d_flip_,
                       input->getData(ReSolve::memory::DEVICE), 
                       output->getData(ReSolve::memory::DEVICE));
    mem_.deviceSynchronize();


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

    mem_.allocateArrayOnDevice(&d_labels_, n_); 
    mem_.allocateArrayOnDevice(&d_flip_, n_); 

    //then copy

    mem_.copyArrayHostToDevice(d_labels_, h_labels_, n_);
    mem_.copyArrayHostToDevice(d_flip_, h_flip_, n_);

    mem_.deviceSynchronize();
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
    mem_.copyArrayHostToDevice(d_labels_, h_labels_, n_);
    mem_.copyArrayHostToDevice(d_flip_, h_flip_, n_);

    mem_.deviceSynchronize();
    return 0;
  }
}
