#include "RandSketchingFWHT.hpp"
#include <resolve/MemoryUtils.hpp>
#include <resolve/vector/Vector.hpp>
#include <math.h>
#ifdef RESOLVE_USE_HIP
#include <resolve/hip/hipKernels.h>
#endif
#ifdef RESOLVE_USE_CUDA
#include <resolve/cuda/cudaKernels.h>
#endif
#include <resolve/RandSketchingFWHT.hpp> 
namespace ReSolve 
{
  RandSketchingFWHT::RandSketchingFWHT()
  {
    h_seq_ = nullptr;
    h_D_ = nullptr;
    h_perm_ = nullptr;

    d_D_ = nullptr;
    d_perm_ = nullptr;
    d_aux_ = nullptr; 
  }

  // destructor
  RandSketchingFWHT::~RandSketchingFWHT()
  {
    delete h_seq_;
    delete h_D_;
    delete h_perm_;

    mem_.deleteOnDevice(d_D_);
    mem_.deleteOnDevice(d_perm_);
    mem_.deleteOnDevice(d_aux_);
  }

  // Actual sketching process
  int RandSketchingFWHT::Theta(vector_type* input, vector_type* output)
  {
    mem_.setZeroArrayOnDevice(d_aux_, N_);
    FWHT_scaleByD(n_, 
                  d_D_,
                  input->getData(ReSolve::memory::DEVICE), 
                  d_aux_);  

    mem_.deviceSynchronize();
    FWHT(1, log2N_, d_aux_);

    mem_.deviceSynchronize();
    FWHT_select(k_rand_, 
                d_perm_, 
                d_aux_, 
                output->getData(ReSolve::memory::DEVICE)); 
    mem_.deviceSynchronize();
    // remember - scaling is the solver's problem 
    return 0;
  }

  // Setup the parameters, sampling matrices, permuations, etc
  int RandSketchingFWHT::setup(index_type n, index_type k)
  {
    k_rand_ = k;
    n_ = n;
    // pad to the nearest power of 2
    N_ = pow(2, ceil(log(n_)/log(2)));
    log2N_ = log2(N_);
    one_over_k_ = 1.0/sqrt((real_type) k_rand_);

    srand(time(NULL)); 

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

    mem_.allocateArrayOnDevice(&d_perm_, k_rand_); 
    mem_.allocateArrayOnDevice(&d_D_, n_); 
    mem_.allocateArrayOnDevice(&d_aux_, N_); 

    //then copy

    mem_.copyArrayHostToDevice(d_perm_, h_perm_, k_rand_);
    mem_.copyArrayHostToDevice(d_D_, h_D_, n_);

    return 0;
  }

  //to be fixed, this can be done on the GPU
  int RandSketchingFWHT::reset() // if needed can be reset (like when Krylov method restarts)
  {
    srand(time(NULL)); 

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

    //and copy

    mem_.copyArrayHostToDevice(d_perm_, h_perm_, k_rand_);
    mem_.copyArrayHostToDevice(d_D_, h_D_, n_);

    return 0;
  }
}
