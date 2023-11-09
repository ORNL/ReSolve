#include <resolve/matrix/Sparse.hpp>
#include <resolve/vector/Vector.hpp>

#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/utilities/logger/Logger.hpp>

#include "SystemSolver.hpp"


namespace ReSolve
{
  // Create a shortcut name for Logger static class
  using out = io::Logger;

  SystemSolver::SystemSolver()
  {
    //set defaults:
    factorizationMethod = "klu";
    refactorizationMethod = "klu";
    solveMethod = "klu";
    IRMethod = "none";
    
    this->setup();
  }
  SystemSolver::~SystemSolver()
  {
    //delete the matrix and all the solvers and all their workspace
  }

  int SystemSolver::setMatrix(matrix::Sparse* A)
  {
    A_ = A;
    return 0;
  }

  int SystemSolver::setup()
  {
    if (factorizationMethod == "klu") {
      KLU_ = new ReSolve::LinSolverDirectKLU();
      KLU_->setupParameters(1, 0.1, false);
    }
    return 0;
  }

  int SystemSolver::analyze()
  {
    if (A_ == nullptr) {
      out::error() << "System matrix not set!\n";
      return 1;
    }

    if (factorizationMethod == "klu") {
      KLU_->setup(A_);
      return KLU_->analyze();
    } 
    return 1;  
  }

  int SystemSolver::factorize()
  {
    if (factorizationMethod == "klu") {
      return KLU_->factorize();
    } 
    return 1;
  }

  int SystemSolver::refactorize()
  {
    if (factorizationMethod == "klu") {
      return KLU_->refactorize();
    } 
    return 1;
  }

  int SystemSolver::solve(vector_type* x, vector_type* rhs)
  {
    if (factorizationMethod == "klu") {
      return KLU_->solve(x, rhs);
    } 
    return 1;
  }

  int SystemSolver::refine(vector_type* x, vector_type* rhs)
  {
    return 1;
  }

} // namespace ReSolve
