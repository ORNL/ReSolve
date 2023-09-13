#pragma once
#include "Common.hpp"
#include <resolve/vector/VectorHandler.hpp>
#include <iostream>
#include <cassert>
namespace ReSolve 
{
  enum GS_variant { mgs, cgs2, mgs_two_synch, mgs_pm, cgs};
  class GramSchmidt
  {
    using vector_type = vector::Vector;
    public:
    GramSchmidt();
    GramSchmidt(VectorHandler* vh, GS_variant variant);
    ~GramSchmidt();
    int setVariant(GS_variant variant);
    GS_variant  getVariant();
    real_type* getL(); //only for low synch, returns null ptr otherwise 

    int setup(index_type n, index_type restart);
    int orthogonalize(index_type n, vector_type* V, real_type* H, index_type i, std::string memspace);

    private:
    
    GS_variant variant_;
    bool setup_complete_; //to avoid double allocations and stuff

    index_type num_vecs_; //the same as restart  
    vector_type* vec_rv_;
    vector_type* vec_Hcolumn_;
//    vector_type* d_H_col_;

    real_type* h_L_;
    real_type* h_rv_;
    real_type* h_aux_;
    VectorHandler* vector_handler_;

    vector_type* vec_v_; // aux variable
    vector_type* vec_w_; // aux variable
  };

}//namespace
