/**
 * @file LinSolverIterativeRandFGMRES.hpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @brief Declaration of LinSolverIterativeRandFGMRES class
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
   * @brief Randomized (F)GMRES
   *
   * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
   *
   * @note Pointers to MatrixHandler and VectorHandler objects are inherited from
   * LinSolver base class.
   *
   */
  class LinSolverIterativeRandFGMRES : public LinSolverIterative
  {
  private:
    using vector_type = vector::Vector;

  public:
    enum SketchingMethod
    {
      cs = 0, // count sketch
      fwht
    }; // fast Walsh-Hadamard transform

    LinSolverIterativeRandFGMRES(MatrixHandler*  matrix_handler,
                                 VectorHandler*  vector_handler,
                                 SketchingMethod rand_method,
                                 GramSchmidt*    gs);

    LinSolverIterativeRandFGMRES(index_type      restart,
                                 real_type       tol,
                                 index_type      maxit,
                                 index_type      conv_cond,
                                 MatrixHandler*  matrix_handler,
                                 VectorHandler*  vector_handler,
                                 SketchingMethod rand_method,
                                 GramSchmidt*    gs);

    ~LinSolverIterativeRandFGMRES();

    int solve(vector_type* rhs, vector_type* x) override;
    int setup(matrix::Sparse* A) override;
    int resetMatrix(matrix::Sparse* new_A) override;
    int setupPreconditioner(std::string name, LinSolverDirect* LU_solver) override;
    int setOrthogonalization(GramSchmidt* gs) override;

    int        setRestart(index_type restart);
    int        setFlexible(bool is_flexible);
    int        setConvergenceCondition(index_type conv_cond);
    index_type getRestart() const;
    index_type getConvCond() const;
    bool       getFlexible() const;

    index_type getKrand();
    int        setSketchingMethod(SketchingMethod method);

    int         setCliParam(const std::string id, const std::string value) override;
    std::string getCliParamString(const std::string id) const override;
    index_type  getCliParamInt(const std::string id) const override;
    real_type   getCliParamReal(const std::string id) const override;
    bool        getCliParamBool(const std::string id) const override;
    int         printCliParam(const std::string id) const override;

  private:
    enum ParamaterIDs
    {
      TOL = 0,
      MAXIT,
      RESTART,
      CONV_COND,
      FLEXIBLE
    };

    index_type restart_{10};    ///< GMRES restart
    index_type conv_cond_{2};   ///< GMRES convergence condition
    bool       flexible_{true}; ///< If using flexible GMRES (FGMRES) algorithm

  private:
    int  allocateSolverData();
    int  freeSolverData();
    int  allocateSketchingData();
    int  freeSketchingData();
    void setMemorySpace();
    void initParamList();
    void precV(vector_type* rhs, vector_type* x); ///< Apply preconditioner

    memory::MemorySpace memspace_;

    vector_type* vec_V_{nullptr};
    vector_type* vec_Z_{nullptr};
    // for performing Gram-Schmidt
    vector_type* vec_S_{nullptr}; ///< this is where sketched vectors are stored

    real_type*   h_H_{nullptr};
    real_type*   h_c_{nullptr};
    real_type*   h_s_{nullptr};
    real_type*   h_rs_{nullptr};
    vector_type* vec_aux_{nullptr};

    GramSchmidt*     GS_{nullptr};
    LinSolverDirect* LU_solver_{nullptr};
    index_type       n_{0};
    real_type        one_over_k_{1.0};

    index_type         k_rand_{0}; ///< size of sketch space. We need to know it so we can allocate S!
    MemoryHandler      mem_;       ///< Device memory manager object
    SketchingHandler*  sketching_handler_{nullptr};
    SketchingMethod    sketching_method_;
    memory::DeviceType device_type_{memory::NONE};
    bool               is_solver_set_{false};
    bool               is_sketching_set_{false};
  };
} // namespace ReSolve
