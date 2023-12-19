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
  /**
   * @brief Default constructor
   * 
   * @post All class variables are set to nullptr.
   * 
   * @todo There is little utility for the default constructor. Maybe remove?.
   */
  RandSketchingCountSketch::RandSketchingCountSketch()
  {
    h_labels_ = nullptr;
    h_flip_ = nullptr;

    d_labels_ = nullptr;
    d_flip_ = nullptr;
  }

  // destructor
  /**
   * @brief Default de-constructor
   * 
   */
  RandSketchingCountSketch::~RandSketchingCountSketch()
  {
    delete h_labels_;
    delete h_flip_;
    mem_.deleteOnDevice(d_labels_);
    mem_.deleteOnDevice(d_flip_);
  }

  // Actual sketching process
  /** 
   * @brief Sketching method - it sketches a given vector (shrinks its size)
   *
   * @param[in]  input   - input vector, size _n_ 
   * @param[out] output  - output vector, size _k_ 
   * 
   * @pre both vectors are allocated. Setup function from this class has been called.
   *
   * @return 0 of successful, -1 otherwise (TODO). 
   */
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
  /** 
   * @brief Sketching method setup. This function allocated _labels_, and _flip_ arrays and, populates them.
   *
   * @param[in]  n  - size of base (non-sketched) vector
   * @param[in]  k  - size of sketched vector. 
   * 
   * @post Everything is set up so you call call Theta.
   *
   * @return 0 of successful, -1 otherwise. 
   */
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
  /** 
   * @brief Reset values in the arrays used for sketching. If the solver restarts, call this method between restarts.
   *
   * @post Everything is set up so you call call Theta.
   *
   * @return 0 of successful, -1 otherwise. 
   */
  int RandSketchingCountSketch::reset() // if needed can be reset (like when Krylov method restarts)
  {
    srand(static_cast<unsigned>(time(nullptr)));
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
