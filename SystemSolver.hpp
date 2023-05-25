//this is to solve the system, can call different linear solvers if necessary
namespace resolve
{
  class resolveSystemSolver
  {
    resolveSystemSolver();
    resolveSystemSolver(std::string factorizationMethod, std::string refactorizationMethod, std::string solveMethod, std::string IRMethod);

    ~resolveSystemSolver();

    public:
    analyze(); //    symbolic part
    factorize(); //  numeric part
    refactorize();
    solve(double* x, double* rhs); // for triangular solve
    refine(double, double* rhs); // for iterative refinement

    // we update the matrix once it changed
    updateMatrix(std::string format, int * ia, int *ja, double *a);

    private:
    
    resolveMatrix A;
    std::string factorizationMethod;
    std::string refactorizationMethod;
    std::string solveMethod;
    std::string IRMethod;

    setup();
    //internal function to setup the different solvers. IT IS RUN ONCE THROUGH CONSTRUCTOR.

    // add factorizationSolver, iterativeSolver, triangularSolver

  };
}
