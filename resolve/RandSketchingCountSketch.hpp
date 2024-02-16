#pragma once
#include <resolve/Common.hpp>
#include <resolve/RandSketchingManager.hpp>

namespace ReSolve {

  // Forward declaration of vector::Vector class
  namespace vector
  {
    class Vector;
  }

  class RandSketchingCountSketch : public RandSketchingManager
  {

    using vector_type = vector::Vector;
    public: 

      // constructor
      RandSketchingCountSketch(memory::MemorySpace memspace);

      // destructor
      virtual ~RandSketchingCountSketch();

      // Actual sketching process
      virtual int Theta(vector_type* input, vector_type* output);

      // Setup the parameters, sampling matrices, permuations, etc
      virtual int setup(index_type n, index_type k);
      virtual int reset(); // if needed can be reset (like when Krylov method restarts)

    private:
      index_type* h_labels_;///< label array size _n_, with values from _0_ to _k-1_ assigned by random
      index_type* h_flip_; ///< flip array with valyes of 1 and -1 assigned by random

      index_type* d_labels_; ///< h_labels GPU counterpart
      index_type* d_flip_;   ///< h_flip GPU counterpart
      memory::MemorySpace memspace_;
  };
}
