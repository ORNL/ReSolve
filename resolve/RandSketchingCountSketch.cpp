#include "RandSketchingCountSketch.hpp"

RandSketchingCountSketch::RandSketchingCountSketch()
{
  h_labels_ = nullptr;
  h_flip_ = nullptr;
  
  d_labels_ = nullptr;
  d_flip_ = nullptr;
}

// destructor
virtual RandSketchingCountSketch::~RandSketchingCountSketch()
{

}

// Actual sketching process
virtual int RandSketchingCountSketch::Theta(vector_type* input, vector_type* output)
{
}

// Setup the parameters, sampling matrices, permuations, etc
virtual int RandSketchingCountSketch::setup(index_type n, index_type k)
{
}

virtual int RandSketchingCountSketch::reset(); // if needed can be reset (like when Krylov method restarts)
{
}

