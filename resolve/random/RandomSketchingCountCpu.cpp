#include <resolve/vector/Vector.hpp>
#include <resolve/random/cpuSketchingKernels.h>
#ifdef RESOLVE_USE_HIP
#include <resolve/hip/hipSketchingKernels.h>
#endif
#ifdef RESOLVE_USE_CUDA
#include <resolve/cuda/cudaSketchingKernels.h>
#endif
#include <resolve/random/RandomSketchingCountCpu.hpp> 

namespace ReSolve 
{
  /**
   * @brief Default constructor
   * 
   * @post All class variables set to nullptr.
   * 
   * @todo Consider removing.
   */
  RandomSketchingCountCpu::RandomSketchingCountCpu()
  {
  }

  /// Destructor
  RandomSketchingCountCpu::~RandomSketchingCountCpu()
  {
    delete [] h_labels_;
    delete [] h_flip_;
  }

  /**
   * @brief Sketching method using CountSketch algorithm.
   * 
   * Implements actual sketching process.
   *
   * @param[in]  input - Vector size _n_
   * @param[out]  output - Vector size _k_ 
   *
   * @pre Both input and output variables are initialized and of correct size.
   * Setup has been run at least once 
   * 
   * @return output = Theta (input) 
   * 
   */
  int RandomSketchingCountCpu::Theta(vector_type* input, vector_type* output)
  {
    cpu::count_sketch_theta(n_,
                            k_rand_,
                            h_labels_,
                            h_flip_,
                            input->getData(memory::HOST),
                            output->getData(memory::HOST));
    return 0;
  }

  /**
   * @brief Sketching setup method for CountSketch algorithm.
   * 
   * Sets up parameters, sampling matrices, permuations, etc.
   * 
   * @param[in]  n - Size of base vector
   * @param[in]  k - Size of sketch 
   *
   * @pre _n_ > _k_. 
   * 
   * @post The arrays needed for performing sketches with CountSketch (_flip_ and _labels_ )
   * are initialized. If GPU is enabled, the arrays will be copied to the GPU, as well 
   * 
   */
  int RandomSketchingCountCpu::setup(index_type n, index_type k)
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

    return 0;
  }

  /**
   * @brief Reset CountSketch arrays (for intance, if solver restarted)
   *
   * @param[in]  n - Size of base vector
   * @param[in]  k - Size of sketch 
   *
   * @pre _n_ > _k_. _k_ value DID NOT CHANGE from the time the setup function
   * was executed.
   * 
   * @post The arrays needed for performing sketches with CountSketch
   * (_flip_ and _labels_ ) are reset to new values. If GPU is enabled, the
   * arrays will be copied to the GPU, as well 
   * 
   * @todo Need to be fixed, this can be done on the GPU.
   */
  int RandomSketchingCountCpu::reset() // if needed can be reset (like when Krylov method restarts)
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

    return 0;
  }
}
