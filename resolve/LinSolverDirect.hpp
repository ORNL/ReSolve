/**
 * @file LinSolverDirect.hpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Declaration of direct solver base class.
 *
 */
#pragma once

#include <string>

#include <resolve/LinSolver.hpp>

namespace ReSolve
{
  class LinSolverDirect : public LinSolver
  {
  public:
    using vector_type = vector::Vector;
    LinSolverDirect();
    virtual ~LinSolverDirect();
    virtual int setup(matrix::Sparse* A   = nullptr,
                      matrix::Sparse* L   = nullptr,
                      matrix::Sparse* U   = nullptr,
                      index_type*     P   = nullptr,
                      index_type*     Q   = nullptr,
                      vector_type*    rhs = nullptr);

    virtual int setupCsr(matrix::Sparse* A,
                         matrix::Sparse* L   = nullptr,
                         matrix::Sparse* U   = nullptr,
                         index_type*     P   = nullptr,
                         index_type*     Q   = nullptr,
                         vector_type*    rhs = nullptr);

    virtual int analyze(); // the same as symbolic factorization
    virtual int factorize();
    virtual int refactorize();
    virtual int solve(vector_type* rhs, vector_type* x) = 0;
    virtual int solve(vector_type* x)                   = 0;

    virtual matrix::Sparse* getLFactorCsr();
    virtual matrix::Sparse* getUFactorCsr();
    virtual vector_type*    getRFactorCsr();
    virtual matrix::Sparse* getLFactor();
    virtual matrix::Sparse* getUFactor();
    virtual index_type*     getPOrdering();
    virtual index_type*     getQOrdering();

  protected:
    matrix::Sparse* L_{nullptr};
    matrix::Sparse* U_{nullptr};
    vector::Vector* R_{nullptr};
    index_type*     P_{nullptr};
    index_type*     Q_{nullptr};
  };

} // namespace ReSolve
