/**
 * @file RandomSketchingCountHip.cpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Definition of RandomSketchingCountHip class.
 *
 */
#include <resolve/hip/hipSketchingKernels.h>

#include <resolve/MemoryUtils.hpp>
#include <resolve/random/RandomSketchingCountHip.hpp>
#include <resolve/vector/Vector.hpp>

namespace ReSolve
{
  /**
   * @brief Default constructor
   *
   * @post All class variables set to nullptr.
   */
  RandomSketchingCountHip::RandomSketchingCountHip()
  {
  }

  /// Destructor
  RandomSketchingCountHip::~RandomSketchingCountHip()
  {
    delete[] h_labels_;
    delete[] h_flip_;
    mem_.deleteOnDevice(d_labels_);
    mem_.deleteOnDevice(d_flip_);
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
   * Setup has been run at least once.
   *
   * @return 0 if successful, !=0 otherwise (TODO).
   *
   */
  int RandomSketchingCountHip::Theta(vector_type* input, vector_type* output)
  {
    mem_.deviceSynchronize();
    hip::count_sketch_theta(n_,
                            k_rand_,
                            d_labels_,
                            d_flip_,
                            input->getData(memory::DEVICE),
                            output->getData(memory::DEVICE));
    mem_.deviceSynchronize();
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
   * @post The arrays needed for performing sketches with CountSketch
   * (_flip_ and _labels_ ) are initialized. If GPU is enabled, the arrays
   * will be copied to the GPU, as well
   *
   * @return 0 if successful, !=0 otherwise (TODO).
   */
  int RandomSketchingCountHip::setup(index_type n, index_type k)
  {
    k_rand_ = k;
    n_      = n;
    srand(static_cast<unsigned>(time(nullptr)));
    // allocate labeling scheme vector and move to GPU

    h_labels_ = new int[n_];
    // allocate sgn - a vector of flip signs
    h_flip_   = new int[n_];

    // populate labeling scheme (can be done on the gpu really)
    // to be fixed, this can be done on the GPU
    for (int i = 0; i < n; ++i)
    {
      h_labels_[i] = rand() % k_rand_;
      int r        = rand() % 100;
      if (r < 50)
      {
        h_flip_[i] = -1;
      }
      else
      {
        h_flip_[i] = 1;
      }
    }

    // allocate arrays on device
    mem_.allocateArrayOnDevice(&d_labels_, n_);
    mem_.allocateArrayOnDevice(&d_flip_, n_);
    // then copy
    mem_.copyArrayHostToDevice(d_labels_, h_labels_, n_);
    mem_.copyArrayHostToDevice(d_flip_, h_flip_, n_);
    mem_.deviceSynchronize();

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
   * @return 0 if successful, !=0 otherwise (TODO).
   *
   * @todo Need to be fixed, this can be done on the GPU.
   */
  int RandomSketchingCountHip::reset() // if needed can be reset (like when Krylov method restarts)
  {
    for (int i = 0; i < n_; ++i)
    {
      h_labels_[i] = rand() % k_rand_;

      int r = rand() % 100;
      if (r < 50)
      {
        h_flip_[i] = -1;
      }
      else
      {
        h_flip_[i] = 1;
      }
    }

    mem_.copyArrayHostToDevice(d_labels_, h_labels_, n_);
    mem_.copyArrayHostToDevice(d_flip_, h_flip_, n_);
    mem_.deviceSynchronize();

    return 0;
  }
} // namespace ReSolve
