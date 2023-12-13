#pragma once
#include "Common.hpp"
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/MemoryUtils.hpp>
#include <iostream>
#include <cassert>
namespace ReSolve 
{
  class GramSchmidt
  {
      using vector_type = vector::Vector;
    public:
      enum GSVariant { mgs = 0, 
                      cgs2 = 1,
                      mgs_two_synch = 2, 
                      mgs_pm = 3, 
                      cgs1 = 4 };

      GramSchmidt();
      GramSchmidt(VectorHandler* vh, GSVariant variant);
      ~GramSchmidt();
      int setVariant(GramSchmidt::GSVariant variant);
      GSVariant  getVariant();
      real_type* getL(); //only for low synch, returns null ptr otherwise 

      int setup(index_type n, index_type restart);
      int orthogonalize(index_type n, vector_type* V, real_type* H, index_type i, memory::MemorySpace memspace);
      bool isSetupComplete();

    private:
    
      GSVariant variant_;
      bool setup_complete_; //to avoid double allocations and stuff

      index_type num_vecs_; //the same as restart  
      vector_type* vec_rv_{nullptr};
      vector_type* vec_Hcolumn_{nullptr};

      real_type* h_L_{nullptr};
      real_type* h_rv_{nullptr};
      real_type* h_aux_{nullptr};
      VectorHandler* vector_handler_{nullptr};

      vector_type* vec_v_{nullptr}; // aux variable
      vector_type* vec_w_{nullptr}; // aux variable
    
      MemoryHandler mem_; ///< Device memory manager object
  };

}//namespace
