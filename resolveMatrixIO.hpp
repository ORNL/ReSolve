// this is for standalone testing (will NOT  be used in hiop)
//
#include "resolveCommon.hpp"
#include "resolveMatrix.hpp"

namespace ReSolve
{
  class resolveMatrixIO{
    public:
      resolveMatrixIO();
      ~resolveMatrixIO();

      resolveMatrix* readMatrixFromFile(std::string filename);
      void readAndUpdateMatrix(std::string filename, resolveMatrix* A);
      resolveReal* readRhsFromFile(std::string filename); 
      resolveReal* readAndUpdateRhs(std::string filename, resolveReal* rhs); 

      resolveInt writeMatrixToFile(resolveMatrix* A, std::string filename); 
      resolveInt writeVectorToFile(resolveReal* x, std::string filename); 
  };
}
