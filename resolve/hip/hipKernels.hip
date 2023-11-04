#include "hipKernels.h"

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

//computes V^T[u1 u2] where v is n x k and u1 and u2 are nx1
template <size_t Tv5 = 1024>
__global__ void MassIPTwoVec_kernel(const double* __restrict__ u1, 
                                    const double* __restrict__ u2, 
                                    const double* __restrict__ v, 
                                    double* result,
                                    const int k, 
                                    const int N)
{
  int t = threadIdx.x;
  int bsize = blockDim.x;

  // assume T threads per thread block (and k reductions to be performed)
  volatile __shared__ double s_tmp1[Tv5];

  volatile __shared__ double s_tmp2[Tv5];
  // map between thread index space and the problem index space
  int j = blockIdx.x;
  s_tmp1[t] = 0.0;
  s_tmp2[t] = 0.0;
  int nn = t;
  double can1, can2, cbn;

  while (nn < N) {
    can1 = u1[nn];
    can2 = u2[nn];

    cbn = v[N * j + nn];
    s_tmp1[t] += can1 * cbn;
    s_tmp2[t] += can2 * cbn;

    nn += bsize;
  }

  __syncthreads();

  if (Tv5 >= 1024) {
    if(t < 512) {
      s_tmp1[t] += s_tmp1[t + 512];
      s_tmp2[t] += s_tmp2[t + 512];
    }
    __syncthreads();
  }
  if (Tv5 >= 512) {
    if(t < 256) {
      s_tmp1[t] += s_tmp1[t + 256];
      s_tmp2[t] += s_tmp2[t + 256];
    }
    __syncthreads();
  }
  {
    if (t < 128) {
      s_tmp1[t] += s_tmp1[t + 128];
      s_tmp2[t] += s_tmp2[t + 128];
    }
    __syncthreads();
  }
  {
    if (t < 64) {
      s_tmp1[t] += s_tmp1[t + 64];
      s_tmp2[t] += s_tmp2[t + 64];
    }
    __syncthreads();
  }

  if (t < 32) {
    s_tmp1[t] += s_tmp1[t + 32];
    s_tmp2[t] += s_tmp2[t + 32];

    s_tmp1[t] += s_tmp1[t + 16];
    s_tmp2[t] += s_tmp2[t + 16];

    s_tmp1[t] += s_tmp1[t + 8];
    s_tmp2[t] += s_tmp2[t + 8];

    s_tmp1[t] += s_tmp1[t + 4];
    s_tmp2[t] += s_tmp2[t + 4];

    s_tmp1[t] += s_tmp1[t + 2];
    s_tmp2[t] += s_tmp2[t + 2];

    s_tmp1[t] += s_tmp1[t + 1];
    s_tmp2[t] += s_tmp2[t + 1];
  }
  if (t == 0) {
    result[blockIdx.x] = s_tmp1[0];
    result[blockIdx.x + k] = s_tmp2[0];
  }
}


/// mass AXPY i.e y = y - x*alpha where alpha is [k x 1], needed in 1 and 2 synch GMRES
template <size_t Tmaxk = 1024>
__global__ void massAxpy3_kernel(int N,
                                 int k,
                                 const double* x_data,
                                 double* y_data,
                                 const double* alpha)
{

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int t = threadIdx.x;

  __shared__ double s_alpha[Tmaxk];
  if (t < k) {
    s_alpha[t] = alpha[t];
  }
  __syncthreads();
  while (i < N) {
    double temp = 0.0;
    for(int j = 0; j < k; ++j) {
      temp += x_data[j * N + i] * s_alpha[j];
    }
    y_data[i] -= temp;
    i += (blockDim.x*gridDim.x);
  }
}
__global__ void matrixInfNormPart1(const int n, 
                                   const int nnz, 
                                   const int* a_ia,
                                   const double* a_val, 
                                   double* result)
{
  // one thread per row, pass through rows
  // and sum
  // can be done through atomics
  //\sum_{j=1}^m abs(a_{ij})

  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  while (idx < n) {
    double sum = 0.0;
    for (int i = a_ia[idx]; i < a_ia[idx+1]; ++i) {
      sum = sum + fabs(a_val[i]);
    }
    result[idx] = sum;
    idx += (blockDim.x*gridDim.x);
  }
}


__global__ void permuteVectorP_kernel(const int n, 
                                      const int* perm_vector,
                                      const double* vec_in, 
                                      double* vec_out)
{
  //one thread per vector entry, pass through rows
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  while (idx<n) {
    vec_out[idx] = vec_in[perm_vector[idx]];
    idx += (blockDim.x*gridDim.x);
  }
}

__global__ void permuteVectorQ_kernel(const int n, 
                                      const int* perm_vector,
                                      const double* vec_in, 
                                      double* vec_out)
{
  //one thread per vector entry, pass through rows
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  while (idx<n) {
    vec_out[perm_vector[idx]] = vec_in[idx];
    idx += (blockDim.x*gridDim.x);
  }
}

// for count sketch sketching random method
__global__ void  count_sketch_kernel(const int n,
                                     const int k, 
                                     const int* labels,
                                     const int* flip,
                                     const double* input,
                                     double* output){

  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  while (idx < n){
    //printf("in place %d, I am putting input[perm[%d]] = input[%d] = %f \n", idx,idx, perm[idx], input[perm[idx]] );
    double val = input[idx];
    if (flip[idx] != 1){
      val *= -1.0;
    }    
    atomicAdd(&output[labels[idx]], val);
    idx += blockDim.x * gridDim.x;
  }
}

// for Walsh-Hadamard transform

//kernel 1
__global__ void select_kernel(const int k, 
                              const int* perm,
                              const double* input,
                              double* output){

  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  while (idx < k){
    //printf("in place %d, I am putting input[perm[%d]] = input[%d] = %f \n", idx,idx, perm[idx], input[perm[idx]] );
    output[idx] = input[perm[idx]];
    idx += blockDim.x * gridDim.x;
  }
}

//kernel 2
__global__ void scaleByD_kernel(const int n,
                                const int* D,
                                const double* x,
                                double* y){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 

  while (idx < n){

    if (D[idx] == 1) y[idx]=x[idx];
    else y[idx]= (-1.0)*x[idx];

    idx += blockDim.x * gridDim.x;
  }
}

//kernels 3 and 4


#define ELEMENTARY_LOG2SIZE 11
namespace cg = cooperative_groups;
////////////////////////////////////////////////////////////////////////////////
// Single in-global memory radix-4 Fast Walsh Transform pass
// (for strides exceeding elementary vector size)
////////////////////////////////////////////////////////////////////////////////

__global__ void fwtBatch2Kernel(double* d_Output, double* d_Input, int stride) 
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int N = blockDim.x * gridDim.x * 4;

  double* d_Src = d_Input + blockIdx.y * N;
  double* d_Dst = d_Output + blockIdx.y * N;

  int lo = pos & (stride - 1);
  int i0 = ((pos - lo) << 2) + lo;
  int i1 = i0 + stride;
  int i2 = i1 + stride;
  int i3 = i2 + stride;

  double D0 = d_Src[i0];
  double D1 = d_Src[i1];
  double D2 = d_Src[i2];
  double D3 = d_Src[i3];

  double T;
  T = D0;
  D0 = D0 + D2;
  D2 = T - D2;
  T = D1;
  D1 = D1 + D3;
  D3 = T - D3;
  T = D0;
  d_Dst[i0] = D0 + D1;
  d_Dst[i1] = T - D1;
  T = D2;
  d_Dst[i2] = D2 + D3;
  d_Dst[i3] = T - D3;
}


__global__ void fwtBatch1Kernel(double* d_Output, double* d_Input, int log2N) 
{
  // Handle to thread block group

  cg::thread_block cta = cg::this_thread_block();
  const int N = 1 << log2N;
  const int base = blockIdx.x << log2N;

  //(2 ** 11) * 4 bytes == 8KB -- maximum s_data[] size for G80
  extern __shared__ double s_data[];
  double* d_Src = d_Input + base;
  double* d_Dst = d_Output + base;

  for (int pos = threadIdx.x; pos < N; pos += blockDim.x) {
    s_data[pos] = d_Src[pos];
  }

  // Main radix-4 stages
  const int pos = threadIdx.x;

  for (int stride = N >> 2; stride > 0; stride >>= 2) {
    int lo = pos & (stride - 1);
    int i0 = ((pos - lo) << 2) + lo;
    int i1 = i0 + stride;
    int i2 = i1 + stride;
    int i3 = i2 + stride;

    cg::sync(cta);
    double D0 = s_data[i0];
    double D1 = s_data[i1];
    double D2 = s_data[i2];
    double D3 = s_data[i3];

    double T;
    T = D0;
    D0 = D0 + D2;
    D2 = T - D2;
    T = D1;
    D1 = D1 + D3;
    D3 = T - D3;
    T = D0;
    s_data[i0] = D0 + D1;
    s_data[i1] = T - D1;
    T = D2;
    s_data[i2] = D2 + D3;
    s_data[i3] = T - D3;
  }

  // Do single radix-2 stage for odd power of two
  if (log2N & 1) {

    cg::sync(cta);

    for (int pos = threadIdx.x; pos < N / 2; pos += blockDim.x) {
      int i0 = pos << 1;
      int i1 = i0 + 1;

      double D0 = s_data[i0];
      double D1 = s_data[i1];
      s_data[i0] = D0 + D1;
      s_data[i1] = D0 - D1;
    }
  }

  cg::sync(cta);

  for (int pos = threadIdx.x; pos < N; pos += blockDim.x) {
    d_Dst[pos] = s_data[pos];
  }
}



void mass_inner_product_two_vectors(int n, 
                                    int i, 
                                    double* vec1, 
                                    double* vec2, 
                                    double* mvec, 
                                    double* result)
{
  hipLaunchKernelGGL(MassIPTwoVec_kernel, dim3(i + 1), dim3(1024), 0, 0, vec1, vec2, mvec, result, i + 1, n);
}

void mass_axpy(int n, int i, double* x, double* y, double* alpha)
{
  hipLaunchKernelGGL(massAxpy3_kernel, dim3((n + 384 - 1) / 384), dim3(384), 0, 0, n, i, x, y, alpha);
}

void matrix_row_sums(int n, 
                     int nnz, 
                     int* a_ia,
                     double* a_val, 
                     double* result)
{
  hipLaunchKernelGGL(matrixInfNormPart1, dim3(1000), dim3(1024), 0, 0, n, nnz, a_ia, a_val, result);
}

void permuteVectorP(int n, 
                    int* perm_vector,
                    double* vec_in, 
                    double* vec_out)
{
  hipLaunchKernelGGL(permuteVectorP_kernel, dim3(1000), dim3(1024), 0, 0, n, perm_vector, vec_in, vec_out);
}

void permuteVectorQ(int n, 
                    int* perm_vector,
                    double* vec_in, 
                    double* vec_out)
{
  hipLaunchKernelGGL(permuteVectorQ_kernel, dim3(1000), dim3(1024), 0, 0, n, perm_vector, vec_in, vec_out);
}

void  count_sketch_theta(int n,
                         int k,
                         int* labels,
                         int* flip,
                         double* input,
                         double* output)
{

  hipLaunchKernelGGL(count_sketch_kernel, dim3(10000), dim3(1024), 0, 0, n, k, labels, flip, input, output);
}

void FWHT_select(int k,
                 int* perm,
                 double* input,
                 double* output)
{
  hipLaunchKernelGGL(select_kernel, dim3(1000), dim3(1024), 0, 0,k, perm, input, output);
}

void FWHT_scaleByD(int n,
                   int* D,
                   double* x,
                   double* y)
{
  hipLaunchKernelGGL(scaleByD_kernel, dim3(1000), dim3(1024), 0, 0, n, D, x, y);
}

void FWHT(int M, int log2N, double* d_Data) {

  const int THREAD_N = 1024;
  int N = 1 << log2N;
  dim3 grid((1 << log2N) / (4 * THREAD_N), M, 1);

  for (; log2N > ELEMENTARY_LOG2SIZE; log2N -= 2, N >>= 2, M <<= 2) {
    hipLaunchKernelGGL(fwtBatch2Kernel, grid, dim3(THREAD_N), 0, 0,  d_Data, d_Data, N / 4);
  }

  hipLaunchKernelGGL(fwtBatch1Kernel, dim3(M), dim3(N / 4), N * sizeof(double), 0, d_Data, d_Data, log2N);
}
