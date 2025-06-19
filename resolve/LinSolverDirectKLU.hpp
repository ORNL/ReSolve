#pragma once

#include "Common.hpp"
#include "klu.h"
#include <resolve/LinSolverDirect.hpp>

namespace ReSolve
{
  // Forward declaration of vector::Vector class
  namespace vector
  {
    class Vector;
  }

  // Forward declaration of matrix::Sparse class
  namespace matrix
  {
    class Sparse;
  }

  class LinSolverDirectKLU : public LinSolverDirect
  {
    using vector_type = vector::Vector;

  public:
    LinSolverDirectKLU();
    ~LinSolverDirectKLU();

    int setup(matrix::Sparse* A,
              matrix::Sparse* L   = nullptr,
              matrix::Sparse* U   = nullptr,
              index_type*     P   = nullptr,
              index_type*     Q   = nullptr,
              vector_type*    rhs = nullptr) override;

    int analyze() override; // the same as symbolic factorization
    int factorize() override;
    int refactorize() override;
    int solve(vector_type* rhs, vector_type* x) override;
    int solve(vector_type* x) override;

    matrix::Sparse* getLFactor() override;
    matrix::Sparse* getUFactor() override;
    index_type*     getPOrdering() override;
    index_type*     getQOrdering() override;

    virtual void setPivotThreshold(real_type tol);
    virtual void setOrdering(int ordering);
    virtual void setHaltIfSingular(bool isHalt);

    virtual real_type getMatrixConditionNumber();

    int         setCliParam(const std::string id, const std::string value) override;
    std::string getCliParamString(const std::string id) const override;
    index_type  getCliParamInt(const std::string id) const override;
    real_type   getCliParamReal(const std::string id) const override;
    bool        getCliParamBool(const std::string id) const override;
    int         printCliParam(const std::string id) const override;

  private:
    enum ParamaterIDs
    {
      PIVOT_TOL = 0,
      ORDERING,
      HALT_IF_SINGULAR
    };

    /**
     * @brief Ordering type (during the analysis)
     *
     * Available values are  0 = AMD, 1 = COLAMD, 2 = user provided P, Q.
     *
     * Default is COLAMD.
     */
    int ordering_{1};

    /**
     * @brief Partial pivoing tolerance.
     *
     * If the diagonal entry has a magnitude greater than or equal to tol
     * times the largest magnitude of entries in the pivot column, then the
     * diagonal entry is chosen.
     */
    real_type pivot_threshold_tol_{0.1};

    /**
     * @brief Halt if matrix is singular.
     *
     * If false: keep going. Return a Numeric object with a zero U(k,k).
     * A divide-by-zero may occur when computing L(:,k). The Numeric object
     * can be passed to klu_solve (a divide-by-zero will occur). It can
     * also be safely passed to refactorization methods.
     *
     * If true: stop quickly. klu_factor will free the partially-constructed
     * Numeric object. klu_refactor will not free it, but will leave the
     * numerical values only partially defined.
     */
    bool halt_if_singular_{false};

  private:
    void          initParamList();
    bool          factors_extracted_{false};
    klu_common    Common_; // settings
    klu_symbolic* Symbolic_{nullptr};
    klu_numeric*  Numeric_{nullptr};
  };
} // namespace ReSolve
