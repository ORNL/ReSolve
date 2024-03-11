#pragma once
#include <resolve/Common.hpp>
#include <resolve/random/RandomSketchingImpl.hpp>
#include <resolve/MemoryUtils.hpp>

namespace ReSolve {

  // Forward declaration of vector::Vector class
  namespace vector
  {
    class Vector;
  }

  class RandomSketchingCountCuda : public RandomSketchingImpl
  {

    using vector_type = vector::Vector;
    public: 

      // constructor
      RandomSketchingCountCuda();

      // destructor
      virtual ~RandomSketchingCountCuda();

      // Actual sketching process
      virtual int Theta(vector_type* input, vector_type* output);

      // Setup the parameters, sampling matrices, permuations, etc
      virtual int setup(index_type n, index_type k);
      virtual int reset(); // if needed can be reset (like when Krylov method restarts)

    private:
      index_type n_;      ///< size of base vector
      index_type k_rand_; ///< size of sketched vector

      index_type* h_labels_{nullptr}; ///< label array size _n_, with values from _0_ to _k-1_ assigned by random
      index_type* h_flip_{nullptr};   ///< flip array with values of 1 and -1 assigned by random

      index_type* d_labels_{nullptr}; ///< h_labels GPU counterpart
      index_type* d_flip_{nullptr};   ///< h_flip GPU counterpart

      MemoryHandler mem_; ///< Device memory manager object
  };
}
