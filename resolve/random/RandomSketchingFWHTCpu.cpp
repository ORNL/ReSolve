/**
 * @file RandomSketchingFWHTCpu.cpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Definition of RandomSketchingFWHTCpu class.
 * 
 */
#include <cmath>
#include <limits>
#include <cstring>

#include <resolve/MemoryUtils.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/random/cpuSketchingKernels.h>
#include <resolve/random/RandomSketchingFWHTCpu.hpp> 

namespace ReSolve 
{
  using out = io::Logger;

  /**
   * @brief Default constructor
   * 
   * @post All class variables are set to nullptr.
   */
  RandomSketchingFWHTCpu::RandomSketchingFWHTCpu()
  {
  }

  /**
   * @brief destructor
   * 
   */
  RandomSketchingFWHTCpu::~RandomSketchingFWHTCpu()
  {
    delete [] h_seq_;
    delete [] h_D_;
    delete [] h_perm_;
    delete [] d_aux_;
  }

  /** 
   * @brief Sketching method - it sketches a given vector (shrinks its size)
   * 
   * Implements actual sketching process.
   *
   * @param[in]  input   - input vector, size _n_ 
   * @param[out] output  - output vector, size _k_ 
   * 
   * @pre both vectors are allocated. Setup function from this class has been called.
   * @warning normal FWHT function requires scaling by 1/k. This function does not scale.
   *
   * @return 0 if successful, !=0 otherwise (TODO). 
   */
  int RandomSketchingFWHTCpu::Theta(vector_type* input, vector_type* output)
  {
    std::memset(d_aux_, 0.0, static_cast<size_t>(N_) * sizeof(real_type));
    cpu::FWHT_scaleByD(n_, 
                       h_D_,
                       input->getData(memory::HOST),
                       d_aux_);  

    cpu::FWHT(1, log2N_, d_aux_);

    cpu::FWHT_select(k_rand_, 
                     h_perm_, 
                     d_aux_, 
                     output->getData(memory::HOST));
    return 0;
  }

  /** 
   * @brief Sketching method setup. 
   * 
   * This function allocated P(erm), D (diagonal scaling matrix) and populates
   * them. It also allocates auxiliary arrays.
   *
   * @param[in]  n  - size of base (non-sketched) vector
   * @param[in]  k  - size of sketched vector. 
   * 
   * @post Everything is set up so you can call Theta.
   *
   * @return 0 if successful, !=0 otherwise (TODO). 
   */
  int RandomSketchingFWHTCpu::setup(index_type n, index_type k)
  {
    k_rand_ = k;
    n_ = n;
    // pad to the nearest power of 2
    real_type N_real = std::pow(2.0, std::ceil(std::log(n_)/std::log(2.0)));
    if (N_real > static_cast<real_type>(std::numeric_limits<index_type>::max())) {
      out::error() << "Exceeded numerical limits of index_type ...\n";
      return 1;
    }
    N_ = static_cast<index_type>(N_real);
    log2N_ = static_cast<index_type>(std::log2(N_real));
    one_over_k_ = 1.0/std::sqrt(static_cast<real_type>(k_rand_));

    srand(static_cast<unsigned>(time(nullptr)));

    h_seq_  = new int[N_];
    h_perm_  = new int[k_rand_];
    h_D_  = new int[n_];

    int r;
    int temp;

    for (int i = 0; i < N_; ++i) {
      h_seq_[i] = i;
    } 
    //Fisher-Yates
    for (int i = N_ - 1; i > 0; i--) {
      r = rand() % i; 
      temp = h_seq_[i];
      h_seq_[i] = h_seq_[r];
      h_seq_[r] = temp; 
    }
    for (int i = 0; i < k_rand_; ++i) {
      h_perm_[i] = h_seq_[i];
    }

    // and D
    for (int i = 0; i < n_; ++i){
      r = rand() % 100;
      if (r < 50){
        h_D_[i] = -1;
      } else { 
        h_D_[i] = 1;
      }
    }

    // Allocate auxiliary data array
    d_aux_  = new real_type[N_];

    return 0;
  }

  /** 
   * @brief Reset values in the arrays used for sketching.
   * 
   * Sketching can be reset, similar to Krylov method restarts.
   * If the solver restarts, call this method between restarts.
   *
   * @post Everything is set up so you can call Theta.
   *
   * @return 0 if successful, !=0 otherwise (TODO). 
   * 
   * @todo Need to be fixed, this can be done on the GPU.
   */
  int RandomSketchingFWHTCpu::reset()
  {
    srand(static_cast<unsigned>(time(nullptr)));

    int r;
    int temp;

    for (int i = 0; i < N_; ++i) {
      h_seq_[i] = i;
    }

    //Fisher-Yates
    for (int i = N_ - 1; i > 0; i--) {
      r = rand() % i; 
      temp = h_seq_[i];
      h_seq_[i] = h_seq_[r];
      h_seq_[r] = temp; 
    }
    for (int i = 0; i < k_rand_; ++i) {
      h_perm_[i] = h_seq_[i];
    }

    // and D
    for (int i = 0; i < n_; ++i) {
      r = rand() % 100;
      if (r < 50) {
        h_D_[i] = -1;
      } else { 
        h_D_[i] = 1;
      }
    }

    return 0;
  }
}
