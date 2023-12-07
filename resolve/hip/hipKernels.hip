#include "hipKernels.h"

#include <hip/hip_runtime.h>

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
  //\sum_{j=1}^m fabs(a_{ij})

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

__global__ void vectorInfNorm(const int n, 
                              const double* input, 
                              double* result)
{

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  volatile __shared__ double s_max[1024];
  int t = threadIdx.x;
  int bsize = blockDim.x;
  double local_max = 0.0;
  if (idx < n) {
    local_max = fabs(input[idx]);
  }

  idx += (blockDim.x*gridDim.x);
  
  while (idx < n) {
    local_max = fmax(fabs(input[idx]), local_max);
    idx += (blockDim.x*gridDim.x);
  }
  s_max[t] = local_max;
  __syncthreads();

  // reduction

  if (bsize >= 1024) {
    if(t < 512) {
      s_max[t] = fmax(s_max[t], s_max[t + 512]);
    }
    __syncthreads();
  }
  if (bsize >= 512) {
    if(t < 256) {
      s_max[t] = fmax(s_max[t], s_max[t + 256]);
    }
    __syncthreads();
  }

  if (bsize >= 256) {
    if(t < 128) {
      s_max[t] = fmax(s_max[t], s_max[t + 128]);
    }
    __syncthreads();
  }

  if (bsize >= 128) {
    if(t < 64) {
      s_max[t] = fmax(s_max[t], s_max[t + 64]);
    }
    __syncthreads();
  }
  //unroll for last warp
  if (t < 64) {
    s_max[t] = fmax(s_max[t], s_max[t + 64]);
    s_max[t] = fmax(s_max[t], s_max[t + 32]);
    s_max[t] = fmax(s_max[t], s_max[t + 16]);
    s_max[t] = fmax(s_max[t], s_max[t + 8]);
    s_max[t] = fmax(s_max[t], s_max[t + 4]);
    s_max[t] = fmax(s_max[t], s_max[t + 2]);
    s_max[t] = fmax(s_max[t], s_max[t + 1]);

  }
  if (t == 0) {
    int bid = blockIdx.x;
    int gid = gridDim.x;
    result[blockIdx.x] = s_max[0];
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
    idx+= (blockDim.x*gridDim.x);
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
    idx+= (blockDim.x*gridDim.x);
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


void vector_inf_norm(int n,  
                     double* input,
                     double* buffer, 
                     double* result)
{
  hipLaunchKernelGGL(vectorInfNorm, dim3(1024), dim3(1024), 0, 0, n, input, buffer);
  hipDeviceSynchronize();
  hipLaunchKernelGGL(vectorInfNorm, dim3(1), dim3(1024), 0, 0, 1024, buffer, buffer);
  hipMemcpy(result, buffer, sizeof(double) * 1, hipMemcpyDeviceToHost);
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
