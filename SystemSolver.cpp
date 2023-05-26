namespace 
{
  SystemSolver::SystemSolver(){
  //set defaults:
  factorizationMethod = "klu";
  refactorizationMethod = "glu";
  solveMethod = "glu";
  IRMethod = "none";
  
  this->setup();
  }
  SystemSolver::~SystemSoler()
  {
  //delete the matrix and all the solvers and all their workspace
  
  }

  SystemSolver::setup(){
    if (factorizationMethod == "klu"){
       
    }
  }

  SystemSolver::analyze()
  {
    if (factorizationMethod == "klu"){
    //call klu_analyze
    } 
  
  }
}
