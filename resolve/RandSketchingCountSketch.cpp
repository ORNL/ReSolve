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
  // printf("Setting up theta2: k_rand = %d \n", k_rand);
  srand(time(NULL)); 
  //allocate labeling scheme vector and move to GPU

  h_labels  = new int[n_];
  //allocate sgn - a vector of flip signs
  int* h_flip  = (int*) calloc(n, sizeof(int));

  //populate labeling scheme (can be done on the gpu really)
  for (int i=0; i<n; ++i) {
    h_labels[i] = rand() % k_rand;
    //printf("Label[%d] = %d \n", i, h_labels[i]);
    int r = rand()%100;
    if (r < 50){
      h_flip[i] = -1;
    } else { 
      h_flip[i] = 1;
    }
  }


  checkCudaErrors(cudaMalloc(&d_labels, n * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_flip, n * sizeof(double)));

  //then copy
  checkCudaErrors(cudaMemcpy(d_labels, h_labels, sizeof(int) * n, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_flip, h_flip, sizeof(int) * n, cudaMemcpyHostToDevice));
  free(h_labels);
  free(h_flip);
}

virtual int RandSketchingCountSketch::reset(); // if needed can be reset (like when Krylov method restarts)
{

}

