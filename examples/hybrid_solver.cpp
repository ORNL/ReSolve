#include <iomanip>
#include <iostream>
#include <string>
#include <cstdlib>     // For atoi
#include <fstream>     // For std::ifstream
#include <cmath>       // For sqrt, std::isnan
#include <stdexcept>   // For std::runtime_error
#include <filesystem>  // For std::filesystem::exists and create_directories (C++17)
#include <sstream>     // For std::ostringstream (for formatting file IDs)
#include <chrono>

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
#include "ExampleHelper.hpp"


// Boiler-plate for timing data
struct GPUTimer {
    cudaEvent_t start, stop;

    GPUTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GPUTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void start_timer() { cudaEventRecord(start); }

    float stop_timer() {
        float ms = 0;
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};


// Using namespace for convenience
using namespace ReSolve::constants;

int main(int argc, char* argv[])
{
    // Use the same data types as those you specified in ReSolve build.
    using index_type  = ReSolve::index_type;
    using real_type   = ReSolve::real_type;
    using vector_type = ReSolve::vector::Vector;
    using namespace ReSolve::examples;

    // --- Validate Command Line Arguments ---
    // Expected arguments:
    // argv[0] : executable name
    // argv[1] : matrixFileName (base path, e.g., "/path/to/matrix_ACTIVSg10k_AC_")
    // argv[2] : rhsFileName (base path, e.g., "/path/to/rhs_ACTIVSg10k_AC_")
    // argv[3] : numSystems (integer N)
    // argv[4] to argv[4 + (2*N) - 1] : fileId and rhsId pairs for N systems
    // argv[4 + (2*N)] : outputDirectory (path, still taken as argument but no files written by this code)

    // Calculate minimum required argc: 1 (exec) + 3 (base names + numSystems) + 1 (outputDir)
    // Plus 2 arguments per system (fileId, rhsId)
    if (argc < 5) { // Minimum 5 arguments: exec, mat_base, rhs_base, num_systems, output_dir
        std::cerr << "Usage: " << argv[0]
                  << " <matrix_base_path> <rhs_base_path> <num_systems> "
                  << "[<matrix_id_0> <rhs_id_0> ... <matrix_id_N-1> <rhs_id_N-1>] "
                  << "<output_directory>" << std::endl;
        std::cerr << "Example (1 system): " << argv[0]
                  << " /path/to/ACTIVSg10k_AC/matrix_ACTIVSg10k_AC_ "
                  << "/path/to/ACTIVSg10k_AC/rhs_ACTIVSg10k_AC_ 1 "
                  << "00 00 " // IDs for system 0
                  << "~/ACOPF_RESULTS/HybridSolver" << std::endl;
        return 1; // Indicate error
    }

    std::string matrixFileName = argv[1];
    std::string rhsFileName    = argv[2];
    index_type numSystems      = atoi(argv[3]);

    // Validate that enough ID pairs are provided for numSystems
    // Total expected args = 4 (initial fixed args) + (2 * numSystems) (ID pairs) + 1 (outputDir)
    if (argc != (4 + (2 * numSystems) + 1)) {
        std::cerr << "Error: Incorrect number of command-line arguments." << std::endl;
        std::cerr << "Expected " << (4 + (2 * numSystems) + 1) << " arguments, but received " << argc << "." << std::endl;
        std::cerr << "Please check numSystems and ensure correct pairs of file IDs are provided." << std::endl;
        return 1; // Indicate error
    }

    std::string outputDirectory = argv[4 + (2 * numSystems)]; // Last argument is output directory

    std::cout << "Matrix base path: [" << matrixFileName << "]" << std::endl;
    std::cout << "RHS base path:    [" << rhsFileName << "]" << std::endl;
    std::cout << "Total number of systems to process: " << numSystems << std::endl;
    std::cout << "Output directory specified: [" << outputDirectory << "]" << std::endl;

    // --- Create Output Directory (Requires C++17) ---
    // This block is kept as it's part of the argument parsing and good practice,
    // even if this specific code doesn't write files into it directly.
    if (!std::filesystem::exists(outputDirectory)) {
        std::cout << "Creating output directory: " << outputDirectory << std::endl;
        if (!std::filesystem::create_directories(outputDirectory)) {
            std::cerr << "Error: Could not create output directory: " << outputDirectory << std::endl;
            return 1;
        }
    } else {
        std::cout << "Output directory already exists: " << outputDirectory << std::endl;
    }


    // --- Declare all pointers (initialized to nullptr) ---
    ReSolve::matrix::Csr* A = nullptr; // Matrix A in CSR format

    ReSolve::LinAlgWorkspaceCUDA* workspace_CUDA = nullptr; // GPU workspace
    ReSolve::MatrixHandler* matrix_handler = nullptr;       // Handler for matrix operations
    ReSolve::VectorHandler* vector_handler = nullptr;       // Handler for vector operations

    real_type* rhs_host_array = nullptr; // Host-side C-array for RHS data from file
    real_type* x_host_array   = nullptr;   // Host-side C-array for solution data

    vector_type* vec_rhs = nullptr; // Device vector for RHS
    vector_type* vec_x   = nullptr;   // Device vector for solution
    vector_type* vec_r   = nullptr;   // Device vector for residual

    ReSolve::GramSchmidt* GS = nullptr; // Gram-Schmidt orthogonalization (for FGMRES)

    // Direct solvers
    ReSolve::LinSolverDirectKLU* KLU    = nullptr;
    ReSolve::LinSolverDirectCuSolverRf* Rf     = nullptr;

    // Iterative solver
    ReSolve::LinSolverIterativeFGMRES* FGMRES = nullptr;

    // Example Helper
    ReSolve::examples::ExampleHelper<ReSolve::LinAlgWorkspaceCUDA>* helper = nullptr;

    int status        = 0;
    int status_refactor = 0; // For CuSolverRf refactorization status

    // --- Initialize all core ReSolve objects in a try-catch block ---
    try {
        workspace_CUDA = new ReSolve::LinAlgWorkspaceCUDA;
        workspace_CUDA->initializeHandles();

        matrix_handler = new ReSolve::MatrixHandler(workspace_CUDA);
        vector_handler = new ReSolve::VectorHandler(workspace_CUDA);

        GS = new ReSolve::GramSchmidt(vector_handler, ReSolve::GramSchmidt::CGS2);

        KLU = new ReSolve::LinSolverDirectKLU;
        Rf  = new ReSolve::LinSolverDirectCuSolverRf();

        FGMRES = new ReSolve::LinSolverIterativeFGMRES(matrix_handler, vector_handler, GS);
        helper = new ExampleHelper<ReSolve::LinAlgWorkspaceCUDA>(*workspace_CUDA);

    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation error during initialization: " << e.what() << std::endl;
        goto cleanup;
    } catch (const std::exception& e) {
        std::cerr << "General error during initialization: " << e.what() << std::endl;
        goto cleanup;
    } catch (...) {
        std::cerr << "Unknown error during initialization." << std::endl;
        goto cleanup;
    }

    { // Timing

     // Creating timing struct
     GPUTimer iterTimer; // refactorization profiler
     GPUTimer setupTimer; // CuSolverRf setup profiler
     GPUTimer solveTimer; // CuSolverRf solve profiler
     std::ofstream statsFile;
     statsFile.open("/home/axs2061/ACOPF_RESULTS/HybridSolverOutput/CuSolverRf_stats.csv");
     statsFile << "Iteration,Residue,Time_ms(refactorization),Time_ms(solver)\n";

    // --- Main loop to process each system ---
    for (int i = 0; i < numSystems; ++i)
    {
        // Get file IDs from command line arguments
        index_type arg_idx = 4 + i * 2;
        std::string fileId = argv[arg_idx];
        std::string rhsId  = argv[arg_idx + 1];

        std::string matrixFileNameFull = matrixFileName + fileId + ".mtx";
        std::string rhsFileNameFull    = rhsFileName + rhsId + ".mtx";
        // Removed: std::string solutionFileNameFull = outputDirectory + "/solution_" + fileId + ".mtx";

        std::cout << "\n\n\n========================================================================================================================" << std::endl;
        std::cout << "Processing System " << i << " (ID: " << fileId << ")" << std::endl;
        std::cout << "Reading matrix: [" << matrixFileNameFull << "]" << std::endl;
        std::cout << "Reading RHS:    [" << rhsFileNameFull << "]" << std::endl;
        std::cout << "========================================================================================================================" << std::endl << std::endl;

        std::ifstream mat_file(matrixFileNameFull);
        if (!mat_file.is_open()) {
            std::cerr << "Failed to open matrix file: " << matrixFileNameFull << "\n";
            goto cleanup;
        }
        std::ifstream rhs_file(rhsFileNameFull);
        if (!rhs_file.is_open()) {
            std::cerr << "Failed to open RHS file: " << rhsFileNameFull << "\n";
            mat_file.close();
            goto cleanup;
        }

        bool is_expand_symmetric = true;

        if (i == 0) // First system: create and allocate all necessary objects
        {
            A = ReSolve::io::createCsrFromFile(mat_file, is_expand_symmetric);
            rhs_host_array = ReSolve::io::createArrayFromFile(rhs_file);
            x_host_array = new real_type[A->getNumRows()];

            vec_rhs = new vector_type(A->getNumRows());
            vec_x   = new vector_type(A->getNumRows());
            vec_r   = new vector_type(A->getNumRows());

            vec_x->allocate(ReSolve::memory::HOST);
            vec_x->allocate(ReSolve::memory::DEVICE);
            vec_x->setToZero(ReSolve::memory::HOST);
            vec_x->setToZero(ReSolve::memory::DEVICE);
        }
        else // Subsequent systems: update existing structures
        {
            ReSolve::io::updateMatrixFromFile(mat_file, A);
            ReSolve::io::updateArrayFromFile(rhs_file, &rhs_host_array);
        }

        A->syncData(ReSolve::memory::DEVICE);

        std::cout << "Finished reading the matrix and rhs, size: " << A->getNumRows() << " x " << A->getNumColumns()
                  << ", nnz: " << A->getNnz()
                  << ", symmetric? " << A->symmetric()
                  << ", Expanded? " << A->expanded() << std::endl;
        mat_file.close();
        rhs_file.close();

        // Update host and device data for RHS vector
        vec_rhs->copyDataFrom(rhs_host_array, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
        std::cout << "CSR matrix loaded. Expanded NNZ: " << A->getNnz() << std::endl;

        // --- Solver Logic ---
        if (i < 2) // For the first two systems (i=0, i=1), perform full KLU factorization
        {
            std::cout << "DEBUG: System " << i << ": Performing initial KLU factorization and solve." << std::endl;
	    auto cpu_start = std::chrono::high_resolution_clock::now(); // Profiling klu->analyze()
            KLU->setup(A);
            status = KLU->analyze();
	    auto cpu_end = std::chrono::high_resolution_clock::now();
            std::cout << "KLU analysis status: " << status << std::endl;

	    double klu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
	    std::cout << "Profiler: KLU->setup() and KLU->analyze() costs in ms: " << klu_time_ms << std::endl;

            status = KLU->factorize();
            std::cout << "KLU factorization status: " << status << std::endl;
            status = KLU->solve(vec_rhs, vec_x);
            std::cout << "KLU solve status: " << status << std::endl;

            helper->resetSystem(A, vec_rhs, vec_x);
            // Calculate residual after KLU solve
            std::cout << "Relative residual after KLU->solve(): " << helper->getNormRelativeResidual() << std::endl;

            // Setup CuSolverRf and FGMRES preconditioner after the first system (i=0)
            if (i > 0) {
		// Extract factors and configure refactorization solver
		ReSolve::matrix::Csr* L = (ReSolve::matrix::Csr*) KLU->getLFactor();
		ReSolve::matrix::Csr* U = (ReSolve::matrix::Csr*) KLU->getUFactor();
		if (L == nullptr || U == nullptr)
        	{
          	    std::cout << "Factor extraction from KLU failed!\n";
        	}
		index_type* P = KLU->getPOrdering();
		index_type* Q = KLU->getQOrdering();

		setupTimer.start_timer();
		Rf->setup(A, L, U, P, Q, vec_rhs);
		float setupTimeMs = setupTimer.stop_timer();
		std::cout << "Profiler: CuSolverRf->setup() cost in ms: " << setupTimeMs << std::endl;
            }

            std::cout << "DEBUG: CSR conversion + CuSolverRf setup complete! " << std::endl;

            FGMRES->setRestart(1000);
            FGMRES->setMaxit(2000);
            FGMRES->setup(A);
            FGMRES->setupPreconditioner("LU", Rf); // Set Rf as preconditioner for FGMRES

            // Print FGMRES summary
	    helper->printIrSummary(FGMRES);
        }
        else // if (i >= 1) -- Use CuSolverRf for refactorization, then FGMRES, with KLU redo logic
        {
            std::cout << "DEBUG: System " << i << ": Using CuSolverRf refactorization and FGMRES." << std::endl;

	    iterTimer.start_timer();
            status_refactor = Rf->refactorize(); // Attempt CuSolverRf refactorization
	    float elapsed = iterTimer.stop_timer();

            std::cout << "CuSolverRf refactorization status: " << status_refactor << std::endl;

            // --- Hybrid Logic: Redo KLU if residual is too large or CuSolverRf crashed ---
            if (status_refactor != 0)
            {
                std::cout << "\n \t !!! ALERT !!! cuSolverRf crashed; redoing KLU symbolic and numeric factorization. !!! ALERT !!! \n \n";

                // Redo KLU factorization
	        auto cpu_start = std::chrono::high_resolution_clock::now();
                KLU->setup(A);
                status = KLU->analyze();
	        auto cpu_end = std::chrono::high_resolution_clock::now();
                std::cout << "KLU analysis status (redo): " << status << std::endl;

		double klu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
		std::cout << "Profiler: KLU->setup() and KLU->analyze() costs in ms: " << klu_time_ms << std::endl;

                status = KLU->factorize();
                std::cout << "KLU factorization status (redo): " << status << std::endl;
                status = KLU->solve(vec_rhs, vec_x); // Solve with new KLU factors
                std::cout << "KLU solve status (redo): " << status << std::endl;

		helper->resetSystem(A, vec_rhs, vec_x);

                // Recalculate residual after KLU redo
                std::cout << "Relative residual after KLU->solve(): " << helper->getNormRelativeResidual() << std::endl;

                // Re-setup CuSolverRf with the NEW KLU factors
		ReSolve::matrix::Csr* L = (ReSolve::matrix::Csr*) KLU->getLFactor();
                ReSolve::matrix::Csr* U = (ReSolve::matrix::Csr*) KLU->getUFactor();
                if (L == nullptr || U == nullptr)
                {
                    std::cout << "Factor extraction from KLU failed!\n";
                }
                index_type* P = KLU->getPOrdering();
                index_type* Q = KLU->getQOrdering();

		setupTimer.start_timer();
                Rf->setup(A, L, U, P, Q, vec_rhs);
		float setupTimeMs = setupTimer.stop_timer();
		std::cout << "Profiler: CuSolverRf->setup() cost in ms: " << setupTimeMs << std::endl;
            }
	    else
	    {
		solveTimer.start_timer();
		status = Rf->solve(vec_rhs, vec_x); // Solve with CuSolverRf
                float solve_elapsed = solveTimer.stop_timer();
                std::cout << "CuSolverRf solve status: " << status << std::endl;

		// Calculate residual after CuSolverRf solve
		helper->resetSystem(A, vec_rhs, vec_x);
	        double current_residue = helper->getNormRelativeResidual();
                std::cout << "Relative residual after cuSolverRf->solve(): " << current_residue << std::endl;

                FGMRES->resetMatrix(A); // Reset FGMRES with current matrix A
                FGMRES->solve(vec_rhs, vec_x); // Refine solution with FGMRES
                helper->printIrSummary(FGMRES);

		// Save to file
                statsFile << i << "," << current_residue << "," << elapsed << "," << solve_elapsed << "\n";
		if (i % 10 == 0) statsFile.flush();

	    }

        }

    } // End of for loop

    statsFile.close();
   } // Timing

cleanup: // Central cleanup label for error handling and end of program
    std::cout << "\n--- Cleaning up ReSolve objects ---" << std::endl;

    // Delete pointers only if they were successfully allocated (not nullptr)
    if (A) delete A; A = nullptr;
    if (KLU) delete KLU; KLU = nullptr;
    if (Rf) delete Rf; Rf = nullptr;
    if (helper) delete helper; helper = nullptr;
    if (FGMRES) delete FGMRES; FGMRES = nullptr;
    if (GS) delete GS; GS = nullptr;
    if (x_host_array) delete[] x_host_array; x_host_array = nullptr; // delete[] for C-style array
    if (rhs_host_array) delete[] rhs_host_array; rhs_host_array = nullptr; // delete[] for C-style array
    if (vec_r) delete vec_r; vec_r = nullptr;
    if (vec_x) delete vec_x; vec_x = nullptr;
    if (vec_rhs) delete vec_rhs; vec_rhs = nullptr;
    if (vector_handler) delete vector_handler; vector_handler = nullptr;
    if (matrix_handler) delete matrix_handler; matrix_handler = nullptr;
    if (workspace_CUDA) delete workspace_CUDA; workspace_CUDA = nullptr;

    std::cout << "Cleanup complete. Program exiting." << std::endl;
    return 0; // Return 0 for success, or non-zero if goto was due to error
}

