/**
 * @file SpGEMMImpl.hpp
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief Interface for SpGEMM implementations
 */

#pragma once

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/utilities/logger/Logger.hpp>

namespace ReSolve
{
  using real_type = ReSolve::real_type;

  namespace hykkt
  {
    class SpGEMMImpl
    {
    public:
      SpGEMMImpl()          = default;
      virtual ~SpGEMMImpl() = default;

      virtual void loadProductMatrices(matrix::Csr* A, matrix::Csr* B) = 0;
      virtual void loadSumMatrix(matrix::Csr* D)                       = 0;
      virtual void loadResultMatrix(matrix::Csr** E_ptr)               = 0;

      virtual void compute() = 0;
    };
  } // namespace hykkt
} // namespace ReSolve
