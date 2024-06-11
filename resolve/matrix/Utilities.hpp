#pragma once

#include <functional>
#include <tuple>

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>

namespace ReSolve
{
  namespace matrix
  {
    class Sparse;

    /// @brief Expands a symmetric matrix
    ///
    /// If A is not symmetric or symmetric and expanded this is a no-op.
    /// Otherwise, it will expand the provided symmetric matrix such that all
    /// nonzeroes are stored
    int expand(Sparse* a);

    /// @brief Provides an iterative iterface over a matrix
    ///
    /// Calls to the returned closure will yield pairs where the second element
    /// indicates if there are any more elements left. If there are none, the
    /// first element, a triple (i, j, x), is composed of nullptrs. If there
    /// are elements left, the triple contains elements as x = A(i, j) of the
    /// input matrix A.
    ///
    /// This interfaced is to be used as a bandage. It will be removed in a
    /// later version
    std::function<std::tuple<std::tuple<index_type, index_type, real_type>, bool>()> elements(Sparse*);
  } // namespace matrix
} // namespace ReSolve
