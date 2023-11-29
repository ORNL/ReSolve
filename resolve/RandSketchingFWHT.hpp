#pragma once
#include <resolve/Common.hpp>
#include <resolve/RandSketchingManager.hpp>

namespace ReSolve {
  // Forward declaration of vector::Vector class
  namespace vector
  {
    class Vector;
  }
  
  class RandSketchingFWHT : public RandSketchingManager
  {

    using vector_type = vector::Vector;
    public: 
      // constructor

      RandSketchingFWHT();

      // destructor
      virtual ~RandSketchingFWHT();

      // Actual sketching process
      virtual int Theta(vector_type* input, vector_type* output);

      // Setup the parameters, sampling matrices, permuations, etc
      virtual int setup(index_type n, index_type k);
      virtual int reset(); // if needed can be reset (like when Krylov method restarts)

    private:
      index_type* h_seq_;
      index_type* h_D_;
      index_type* h_perm_;

      index_type* d_D_;
      index_type* d_perm_;
      real_type* d_aux_;
      
      index_type N_;
      index_type log2N_;
      real_type one_over_k_;
  };
}
