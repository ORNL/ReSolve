/**
 * @file LinSolver.cpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Implementation of linear solver base class.
 * 
 */

#include <resolve/matrix/Sparse.hpp>
#include <resolve/utilities/logger/Logger.hpp>

#include "LinSolver.hpp"


namespace ReSolve 
{
  using out = io::Logger;

  LinSolver::LinSolver()
  {
  }

  LinSolver::~LinSolver()
  {
    //destroy the matrix and hadlers
  }

  real_type LinSolver::evaluateResidual()
  {
    //to be implemented
    return 1.0;
  }

  int LinSolver::getParamId(std::string id) const
  {
    auto it = params_list_.find(id);
    if (it == params_list_.end()) {
      out::error() << "Unknown parameter " << id << ".\n";
      return 999;
    }
    return (*it).second;
  }
}



