#include "RandSketchingCountSketch.hpp"
#include <resolve/MemoryUtils.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/cpu/cpuKernels.h>
#ifdef RESOLVE_USE_HIP
#include <resolve/hip/hipKernels.h>
#endif
#ifdef RESOLVE_USE_CUDA
#include <resolve/cuda/cudaKernels.h>
#endif
#include <resolve/random/RandSketchingCountSketch.hpp> 

namespace ReSolve 
{
  /**
   * @brief Default constructor
   * 
   * @post All class variables set to nullptr.
   * 
   * @todo Consider removing.
   */
  RandSketchingCountSketch::RandSketchingCountSketch(memory::MemorySpace memspace)
    : memspace_(memspace)
  {
    std::cout << "Count skecthing memory space: " << memspace_ << "\n";
    h_labels_ = nullptr;
    h_flip_ = nullptr;

    d_labels_ = nullptr;
    d_flip_ = nullptr;
  }

  // destructor
  RandSketchingCountSketch::~RandSketchingCountSketch()
  {
    delete [] h_labels_;
    delete [] h_flip_;
    if (memspace_ == memory::DEVICE) {
      mem_.deleteOnDevice(d_labels_);
      mem_.deleteOnDevice(d_flip_);
    }
  }

  // Actual sketching process
  /**
   * @brief Sketching method using CountSketch algorithm.
   *
   * @param[in]  input - Vector size _n_
   * @param[out]  output - Vector size _k_ 
   *
   * @pre Both input and output variables are initialized and of correct size. Setup has been run at least once 
   * 
   * @return output = Theta (input) 
   * 
   */
  int RandSketchingCountSketch::Theta(vector_type* input, vector_type* output)
  {
    using namespace memory;
   
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
        count_sketch_theta(n_,
                          k_rand_,
                          h_labels_,
                          h_flip_,
                          input->getData(ReSolve::memory::HOST), 
                          output->getData(ReSolve::memory::HOST));
        break;
    }

    return 0;
  }

  /// Setup the parameters, sampling matrices, permuations, etc
  /**
   * @brief Sketching setup method for CountSketch algorithm.
   *
   * @param[in]  n - Size of base vector
   * @param[in]  k - Size of sketch 
   *
   * @pre _n_ > _k_. 
   * 
   * @post The arrays needed for performing sketches with CountSketch (_flip_ and _labels_ ) are initialized. If GPU is enabled, the arrays will be copied to the GPU, as well 
   * 
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
      if (r < 50) {
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
    return 0;
  }

  /// @todo Need to be fixed, this can be done on the GPU.
  /**
   * @brief Reset CountSketch arrays (for intance, if solver restarted)
   *
   * @param[in]  n - Size of base vector
   * @param[in]  k - Size of sketch 
   *
   * @pre _n_ > _k_. _k_ value DID NOT CHANGE from the time the setup function was executed.
   * 
   * @post The arrays needed for performing sketches with CountSketch (_flip_ and _labels_ ) are reset to new values. If GPU is enabled, the arrays will be copied to the GPU, as well 
   * 
   */
  int RandSketchingCountSketch::reset() // if needed can be reset (like when Krylov method restarts)
  {
    for (int i = 0; i < n_; ++i) {
      h_labels_[i] = rand() % k_rand_;

      int r = rand()%100;
      if (r < 50) {
        h_flip_[i] = -1;
      } else { 
        h_flip_[i] = 1;
      }
    }
    if (memspace_ == memory::DEVICE) {
      mem_.copyArrayHostToDevice(d_labels_, h_labels_, n_);
      mem_.copyArrayHostToDevice(d_flip_, h_flip_, n_);

      mem_.deviceSynchronize();
    }
    return 0;
  }
}
