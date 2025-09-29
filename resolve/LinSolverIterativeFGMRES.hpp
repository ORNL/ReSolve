/**
 * @file LinSolverIterativeFGMRES.hpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @brief Declaration of LinSolverIterativeFGMRES class
 *
 */
#pragma once

#include "Common.hpp"
#include <resolve/LinSolverDirect.hpp>
#include <resolve/LinSolverIterative.hpp>
#include <resolve/MemoryUtils.hpp>

namespace ReSolve
{
  // Forward declarations
  class SketchingHandler;
  class GramSchmidt;

  namespace matrix
  {
    class Sparse;
  }

  namespace vector
  {
    class Vector;
  }

  /**
   * @brief (F)GMRES solver
   *
   * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
   *
   * @note MatrixHandler and VectorHandler objects are inherited from
   * LinSolver base class.
   */
  class LinSolverIterativeFGMRES : public LinSolverIterative
  {
    using vector_type = vector::Vector;

  public:
    LinSolverIterativeFGMRES(MatrixHandler* matrix_handler,
                             VectorHandler* vector_handler,
                             GramSchmidt* gs);
    LinSolverIterativeFGMRES(index_type     restart,
                             real_type      tol,
                             index_type     maxit,
                             index_type     conv_cond,
                             MatrixHandler* matrix_handler,
                             VectorHandler* vector_handler,
                             GramSchmidt* gs);
    ~LinSolverIterativeFGMRES();

    int solve(vector_type* rhs, vector_type* x) override;
    int setup(matrix::Sparse* A) override;
    int resetMatrix(matrix::Sparse* new_A) override;
    int setupPreconditioner(std::string name, LinSolverDirect* LU_solver) override;
    int setOrthogonalization(GramSchmidt* gs) override;

    int          setRestart(index_type restart);
    int          setFlexible(bool is_flexible);
    int          setConvergenceCondition(index_type conv_cond);
    index_type   getRestart() const;
    index_type   getConvCond() const;
    bool         getFlexible() const;
    real_type    getEffectiveStability() const; // Getter for the new member variable

    int          setCliParam(const std::string id, const std::string value) override;
    std::string  getCliParamString(const std::string id) const override;
    index_type   getCliParamInt(const std::string id) const override;
    real_type    getCliParamReal(const std::string id) const override;
    bool         getCliParamBool(const std::string id) const override;
    int          printCliParam(const std::string id) const override;

  private:
    enum ParamaterIDs
    {
      TOL = 0,
      MAXIT,
      RESTART,
      CONV_COND,
      FLEXIBLE
    };

    index_type restart_{10};      ///< GMRES restart
    index_type conv_cond_{0};     ///< GMRES convergence condition
    bool       flexible_{true};   ///< If using flexible GMRES (FGMRES) algorithm
    real_type effectiveStability_; ///< Max norm of the preconditioner residual

  private:
    int  allocateSolverData();
    int  freeSolverData();
    void setMemorySpace();
    void initParamList();
    void precV(vector_type* rhs, vector_type* x); ///< Apply preconditioner

    memory::MemorySpace memspace_;

    vector_type* vec_V_{nullptr};
    vector_type* vec_Z_{nullptr};
    vector_type* vec_Y_{nullptr};
    vector_type* vec_R_{nullptr};

    real_type* h_H_{nullptr};
    real_type* h_c_{nullptr};
    real_type* h_s_{nullptr};
    real_type* h_rs_{nullptr};

    GramSchmidt* GS_{nullptr};
    LinSolverDirect* LU_solver_{nullptr};
    index_type     n_{0};
    bool           is_solver_set_{false};

    MemoryHandler mem_; ///< Device memory manager object
  };
} // namespace ReSolve
