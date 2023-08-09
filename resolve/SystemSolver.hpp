//this is to solve the system, can call different linear solvers if necessary
namespace 
{
  class SystemSolver
  {
    SystemSolver();
    SystemSolver(std::string factorizationMethod, std::string refactorizationMethod, std::string solveMethod, std::string IRMethod);

    ~SystemSolver();

    public:
    analyze(); //    symbolic part
    factorize(); //  numeric part
    refactorize();
    solve(double* x, double* rhs); // for triangular solve
    refine(double, double* rhs); // for iterative refinement

    // we update the matrix once it changed
    updateMatrix(std::string format, int * ia, int *ja, double *a);

    private:
    
    Sparse A;
    std::string factorizationMethod;
    std::string refactorizationMethod;
    std::string solveMethod;
    std::string IRMethod;

    setup();
    //internal function to setup the different solvers. IT IS RUN ONCE THROUGH CONSTRUCTOR.

    // add factorizationSolver, iterativeSolver, triangularSolver

  };
}
