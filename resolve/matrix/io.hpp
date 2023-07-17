#include <fstream>
#include <resolve/Vector.hpp>

namespace ReSolve { namespace matrix { namespace io {

  MatrixCOO* readMatrixFromFile(std::istream& file);
  void readAndUpdateMatrix(std::istream& file, MatrixCOO* A);
  real_type* readRhsFromFile(std::istream& file); 
  void readAndUpdateRhs(std::istream& file, real_type** rhs); 

  int writeMatrixToFile(Matrix* A, std::ostream file_out);
  int writeVectorToFile(Vector* vec_x, std::ostream file_out);
}}} // ReSolve::matrix::io