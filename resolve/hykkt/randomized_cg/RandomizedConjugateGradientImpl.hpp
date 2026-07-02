/**
 * @file RandomizedConjugateGradientImpl.hpp
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief Abstract interface for Cholesky Solver implementations
 */

#pragma once
#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>

namespace ReSolve
{
  namespace hykkt
  {
    class RandomizedConjugateGradientImpl
    {
    public:
      RandomizedConjugateGradientImpl()          = default;
      virtual ~RandomizedConjugateGradientImpl() = default;

      virtual int setup(index_type k) = 0;
      virtual int SpMMTallSkinny(matrix::Csr* A, vector::Vector* X, vector::Vector* result) = 0;
      virtual int bestBasis(vector::Vector* R, index_type* h_best_basis, real_type* h_best_basis_norm) = 0;
      virtual int choleskyQr(vector::Vector* W, vector::Vector* R, memory::MemorySpace memspace) = 0;
      virtual int updateW(vector::Vector* W, vector::Vector* L, vector::Vector* B, memory::MemorySpace memspace) = 0;
      virtual int multTSMTTSM(vector::Vector* A, vector::Vector* B, vector::Vector* C, memory::MemorySpace memspace) = 0;
      virtual int updateSSigma(vector::Vector* W, vector::Vector* S, vector::Vector* Zeta, vector::Vector* Sigma, memory::MemorySpace memspace) = 0;
      // virtual int innerProductTSM(vector::Vector* A, vector::Vector* result, memory::MemorySpace memspace) = 0;
    };
  } // namespace hykkt
} // namespace ReSolve
