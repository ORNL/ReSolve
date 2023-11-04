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
      RandSketchingCountSketch();

      // destructor
      virtual ~RandSketchingCountSketch();

      // Actual sketching process
      virtual int Theta(vector_type* input, vector_type* output);

      // Setup the parameters, sampling matrices, permuations, etc
      virtual int setup(index_type n, index_type k);
      virtual int reset(); // if needed can be reset (like when Krylov method restarts)

    private:
      index_type* h_labels_;
      index_type* h_flip_;

      index_type* d_labels_;
      index_type* d_flip_;
  };
}
