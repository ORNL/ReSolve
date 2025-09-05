#include <iomanip>
#include <iostream>
#include <string>
#include <cstdlib>     // For atoi
#include <fstream>     // For std::ifstream
#include <cmath>       // For sqrt, std::isnan
#include <stdexcept>   // For std::runtime_error
#include <filesystem>  // For std::filesystem::exists and create_directories (C++17)
#include <sstream>     // For std::ostringstream (for formatting file IDs)


// ReSolve Library Includes
#include <resolve/GramSchmidt.hpp>
#include <resolve/LinSolverDirectCuSolverRf.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/matrix/io.hpp> // Contains matrix and likely array/vector IO functions
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp> // LinAlgWorkspaceCUDA as per your system setup

// Using namespace for convenience
using namespace ReSolve::constants;

int main(int agrc, char* argv[]) {

    using index_type = ReSolve::index_type;
    using real_type = ReSolve::real_type;
    using vector_type = ReSolve::vector::Vector;

    // --- Validate Command Line Arguments ---
    // Expected arguments:
    // argv[0] : executable name
    // argv[1] : matrixFileName (base path, e.g., "/path/to/matrix_ACTIVSg10k_AC_")

    std::string outputDirectory = argv[4 + (2 * numSystems)]; // Last argument is output directory

   if (argc < 5) { // Minimum 4 arguments: exec, mat_base, num_systems, output_dir
      std::cerr << "Usage: " << argv[0]
                  << " <matrix_base_path> <num_systems> "
                  << "[<matrix_id_0> ...  <matrix_id_N>] "
                  << "<output_directory>" << std::endl;
      std::cerr << "Example (1 system): " << argv[0]
		<< " /path/to/ACTIVSg10k_AC/matrix_ACTIVSg10k_AC_ "
		<< "00" // IDs for system 0
		<< "~/ACOPF_RESULTS/HybridSolver" <<std:endl;
      return 1; // Error


}

   std::string matrixFileName = argv[1];
   index_type numSystems = atoi(argv[2]);


}
