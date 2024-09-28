/**
 * @file ParamTests.hpp
 * @brief Contains definition of ParamTests class.
 * @author Slaven Peles <peless@ornl.org>
 */

#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/GramSchmidt.hpp>
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <tests/unit/TestBase.hpp>

namespace ReSolve { namespace tests {
  /**
   * @brief Class implementing unit tests for Param class.
   * 
   * The ParamTests class is implemented entirely in this header file.
   * Adding new unit test requires simply adding another method to this
   * class.
   */
  class ParamTests : TestBase
  {
  public:
    ParamTests(){}
    virtual ~ParamTests(){}

    TestOutcome paramSetGet()
    {
      TestStatus success;

      success = true;

      index_type restart   = -1;
      real_type  tol       = -1.0;
      index_type maxit     = -1;
      index_type conv_cond = -1;

      LinAlgWorkspaceCpu workspace;
      workspace.initializeHandles();

      MatrixHandler matrix_handler(&workspace);
      VectorHandler vector_handler(&workspace);

      GramSchmidt gs(&vector_handler, GramSchmidt::cgs2);

      // Constructor sets parameters
      LinSolverIterativeFGMRES solver(restart,
                                      tol,
                                      maxit,
                                      conv_cond,
                                      &matrix_handler,
                                      &vector_handler,
                                      &gs);

      index_type restart_out   = 0;
      real_type  tol_out       = 0.0;
      index_type maxit_out     = 0;
      index_type conv_cond_out = 0;
      bool       flexible_out  = false;

      // Use getters to read parameters set by the constructor
      solver.getCliParam("restart",   restart_out  );
      solver.getCliParam("tol",       tol_out      );
      solver.getCliParam("maxit",     maxit_out    );
      solver.getCliParam("conv_cond", conv_cond_out);
      solver.getCliParam("flexible",  flexible_out );

      // Check getters
      success *= (restart == restart_out);
      success *= (maxit == maxit_out);
      success *= (conv_cond == conv_cond_out);
      success *= isEqual(tol, tol_out);
      success *= flexible_out; // Default is flexible = true

      // Pick different parameter values from the input
      std::string restart_in   = "2";
      std::string tol_in       = "2.0";
      std::string maxit_in     = "2";
      std::string conv_cond_in = "2";
      std::string flexible_in  = "no";

      restart   = atoi(restart_in.c_str());
      tol       = atof(tol_in.c_str());
      maxit     = atoi(maxit_in.c_str());
      conv_cond = atoi(conv_cond_in.c_str());

      // Use setters to change FGMRES solver parameters
      solver.setCliParam("restart",   restart_in  );
      solver.setCliParam("tol",       tol_in      );
      solver.setCliParam("maxit",     maxit_in    );
      solver.setCliParam("conv_cond", conv_cond_in);
      solver.setCliParam("flexible",  flexible_in );

      // Read new values
      solver.getCliParam("restart",   restart_out  );
      solver.getCliParam("tol",       tol_out      );
      solver.getCliParam("maxit",     maxit_out    );
      solver.getCliParam("conv_cond", conv_cond_out);
      solver.getCliParam("flexible",  flexible_out );

      // Check setters
      success *= (restart == restart_out);
      success *= (maxit == maxit_out);
      success *= (conv_cond == conv_cond_out);
      success *= isEqual(tol, tol_out);
      success *= !flexible_out; // flexible was set to "no"

      return success.report(__func__);
    }


  private:

  }; // class ParamTests

}} // namespace ReSolve::tests
