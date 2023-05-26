#include "VectorHandler.hpp"
#include <iostream>

namespace ReSolve
{
  VectorHandler::VectorHandler()
  {
  }

  VectorHandler:: VectorHandler(LinAlgWorkspace* new_workspace)
  {
    workspace_ = new_workspace;
  }

  VectorHandler::~VectorHandler()
  {
    //delete the workspace TODO
  }

  Real VectorHandler::dot(Vector* x, Vector* y, std::string memspace)
  { 
    if (memspace == "cuda" ){ 
      LinAlgWorkspaceCUDA* workspaceCUDA = (LinAlgWorkspaceCUDA*) workspace_;
      cublasHandle_t handle_cublas =  workspaceCUDA->getCublasHandle();
      double nrm = 0.0;
      cublasStatus_t st= cublasDdot (handle_cublas,  x->getSize(), x->getData("cuda"), 1, y->getData("cuda"), 1, &nrm);
      if (st!=0) {printf("dot product crashed with code %d \n", st);}
      return nrm;
    } else {
      std::cout<<"Not implemented (yet)"<<std::endl;
      return NAN;
    }
  }
}
