// this is for standalone testing (will NOT  be used in hiop)
//
#include "Common.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"
#include <iomanip>

namespace ReSolve
{
  class MatrixIO{
    public:
      MatrixIO();
      ~MatrixIO();

      Matrix* readMatrixFromFile(std::string filename);
      void readAndUpdateMatrix(std::string filename, Matrix* A);
      Real* readRhsFromFile(std::string filename); 
      Real* readAndUpdateRhs(std::string filename, Real* rhs); 

      Int writeMatrixToFile(Matrix* A, std::string filename); 
      Int writeVectorToFile(Vector* vec_x, std::string filename); 
  };
}
