// this is for standalone testing (will NOT  be used in hiop)
//
#include "resolveMatrix.hpp"
namespace ReSolve
{
  class resolveMatrixIO{
    public:
      resolveMatrixIO();
      ~resolveMatrixIO();

      resolveMatrix* readMatrixFromFile(std::string filename);
      void readAndUpdateMatrix(std::string filename, resolveMatrix* A);
      double* readRhsFromFile(std::string filename); 
      double* readAndUpdateRhs(std::string filename, double* rhs); 

      int writeMatrixToFile(resolveMatrix* A, std::string filename); 
      int writeVectorToFile(double* x, std::string filename); 
  };
}
