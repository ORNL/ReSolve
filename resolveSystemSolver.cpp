namespace resolve
{
  resolveSystemSolver::resolveSystemSolver(){
  //set defaults:
  factorizationMethod = "klu";
  refactorizationMethod = "glu";
  solveMethod = "glu";
  IRMethod = "none";
  
  this->setup();
  }
  resolveSystemSolver::~resolveSystemSoler()
  {
  //delete the matrix and all the solvers and all their workspace
  
  }

  resolveSystemSolver::setup(){
    if (factorizationMethod == "klu"){
       
    }
  }

  resolveSystemSolver::analyze()
  {
    if (factorizationMethod == "klu"){
    //call klu_analyze
    } 
  
  }
}
