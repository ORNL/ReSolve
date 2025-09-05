#include <iomanip>
#include <iostream>
#include <string>
#include <cstdlib>      // For atoi
#include <fstream>      // For std::ifstream
#include <cmath>        // For sqrt, std::isnan
#include <stdexcept>    // For std::runtime_error
#include <filesystem>   // For std::filesystem::exists and create_directories (C++17)
#include <sstream>      // For std::ostringstream (for formatting file IDs)

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

// New include for ExampleHelper utility class
#include "ExampleHelper.hpp"

// Using namespace for convenience
using namespace ReSolve::constants;

int main(int argc, char* argv[])
{
    // Use the same data types as those you specified in ReSolve build.
    using index_type  = ReSolve::index_type;
    using real_type   = ReSolve::real_type;
    using vector_type = ReSolve::vector::Vector;

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
    real_type* x_host_array   = nullptr;  // Host-side C-array for solution data

    vector_type* vec_rhs = nullptr; // Device vector for RHS
    vector_type* vec_x   = nullptr;   // Device vector for solution
    vector_type* vec_residual = nullptr; // Device vector for residual
    vector_type* vec_error = nullptr;  // Device vector for error

    // Removed vec_r, as ExampleHelper will handle the residual vector internally

    ReSolve::GramSchmidt* GS = nullptr; // Gram-Schmidt orthogonalization (for FGMRES)

    // Direct solvers
    ReSolve::LinSolverDirectKLU* KLU      = nullptr;
    ReSolve::LinSolverDirectCuSolverRf* Rf      = nullptr;

    // Iterative solver
    ReSolve::LinSolverIterativeFGMRES* FGMRES = nullptr;

    // ExampleHelper for clean residual calculations and reporting
    ReSolve::examples::ExampleHelper<ReSolve::LinAlgWorkspaceCUDA>* helper = nullptr;

    // Pointers to KLU factors (not owned by main, so not deleted here)
    ReSolve::matrix::Csc* L_csc_klu = nullptr;
    ReSolve::matrix::Csc* U_csc_klu = nullptr;
    index_type* P_klu = nullptr;
    index_type* Q_klu = nullptr;

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

        // Initialize the ExampleHelper, passing the workspace
        helper = new ReSolve::examples::ExampleHelper<ReSolve::LinAlgWorkspaceCUDA>(*workspace_CUDA);

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

        // Moved the print statements to the correct location
        std::cout << "\n\n\n========================================================================================================================" << std::endl;
        std::cout << "Processing System " << i << " (ID: " << fileId << ")" << std::endl;
        std::cout << "Reading: " << matrixFileNameFull << std::endl;
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
	    vec_residual = new vector_type(A->getNumRows());
	    vec_error = new vector_type(A->getNumRows());

            vec_x->allocate(ReSolve::memory::HOST);
            vec_x->allocate(ReSolve::memory::DEVICE);
            vec_x->setToZero(ReSolve::memory::HOST);
            vec_x->setToZero(ReSolve::memory::DEVICE);

	    vec_error->allocate(ReSolve::memory::HOST);
	    vec_error->allocate(ReSolve::memory::DEVICE);
	    vec_error->setToZero(ReSolve::memory::HOST);
	    vec_error->setToZero(ReSolve::memory::DEVICE);
        }
        else // Subsequent systems: update existing structures
        {
            ReSolve::io::updateMatrixFromFile(mat_file, A);
            ReSolve::io::updateArrayFromFile(rhs_file, &rhs_host_array);
        }

        // The matrix `A` is now a valid pointer.
        // We can now safely print its info.
        std::cout << "Finished reading the matrix and rhs, size: " << A->getNumRows() << " x " << A->getNumColumns()
                  << ", nnz: " << A->getNnz()
                  << ", symmetric? " << A->symmetric()
                  << ", Expanded? " << A->expanded() << std::endl;
        std::cout << "Reading RHS:    [" << rhsFileNameFull << "]" << std::endl;
        mat_file.close();
        rhs_file.close();

        // Update host and device data for RHS vector
        vec_rhs->copyDataFrom(rhs_host_array, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
        // Ensure the matrix is also on the device for the GPU-based solvers
        A->syncData(ReSolve::memory::DEVICE);

        std::cout << "CSR matrix loaded. Expanded NNZ: " << A->getNnz() << std::endl;

        // --- Solver Logic ---
        if (i < 1) // For the first system (i=1), perform full KLU factorization
        {
            std::cout << "DEBUG: System " << i << ": Performing initial KLU factorization and solve." << std::endl;
            KLU->setup(A);
            matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);

            status = KLU->analyze();
            std::cout << "KLU analysis status: " << status << std::endl;
            status = KLU->factorize();
            std::cout << "KLU factorization status: " << status << std::endl;
            status = KLU->solve(vec_rhs, vec_x);
            std::cout << "KLU solve status: " << status << std::endl;

            // Use the helper to print the summary after the direct solve
            helper->resetSystem(A, vec_rhs, vec_x);
            helper->printShortSummary();

            // Setup CuSolverRf and FGMRES preconditioner after the first system (i=0)
            if (i == 0) { // Setup Rf and FGMRES preconditioner only once after the first KLU solve
                L_csc_klu = (ReSolve::matrix::Csc*) KLU->getLFactor();
                U_csc_klu = (ReSolve::matrix::Csc*) KLU->getUFactor();
                P_klu = KLU->getPOrdering();
                Q_klu = KLU->getQOrdering();

                // Check if KLU factors are valid before setting up CuSolverRf
                if (L_csc_klu == nullptr || U_csc_klu == nullptr) {
                    std::cerr << "ERROR: KLU factor pointers are null. Aborting CuSolverRf setup.\n";
                    goto cleanup;
                }

                // Setup CuSolverRf with KLU factors directly in CSC format
                Rf->setup(A, L_csc_klu, U_csc_klu, P_klu, Q_klu);
                Rf->refactorize(); // Initial refactorize for Rf
            }

	    FGMRES->setRestart(1000);
            FGMRES->setMaxit(2000);
            FGMRES->setup(A);
            FGMRES->setupPreconditioner("LU", Rf); // Set Rf as preconditioner for FGMRES

	    // Compute the residual: r = b - Ax
            vec_residual->copyDataFrom(vec_rhs, ReSolve::memory::DEVICE, ReSolve::memory::DEVICE);
            matrix_handler->matvec(A, vec_x, vec_residual, &ReSolve::constants::MINUS_ONE, &ReSolve::constants::ONE, ReSolve::memory::DEVICE);

	    std::cout << "DEBUG: Solving error equation with FGMRES." << std::endl;

            // Perform FGMRES solve after initial KLU solve
            FGMRES->solve(vec_residual, vec_error);
	    std::cout << "FGMRES error estimation: " << sqrt(vector_handler->dot(vec_error, vec_error, ReSolve::memory::DEVICE)) << std::endl;

            // Print FGMRES summary using the helper function
            helper->printIrSummary(FGMRES);
            std::cout << "FGMRES Effective Stability: " << FGMRES->getEffectiveStability() << std::endl;

	    // Update the solution: x = x + e
	    std::cout << "DEBUG: Updating solution vector." << std::endl;
	    vector_handler->axpy(&ReSolve::constants::ONE, vec_error, vec_x, ReSolve::memory::DEVICE);

	    // Final residual calculation
	    helper->resetSystem(A, vec_rhs, vec_x);
	    std::cout << "DEBUG: Relative residual after error update: " << helper->getNormRelativeResidual() << std::endl;

	    // Setting vec_error back to zero for future calculations
	    vec_error->setToZero(ReSolve::memory::HOST);
	    vec_error->setToZero(ReSolve::memory::DEVICE);
        }
        else // if (i > 1) -- Use CuSolverRf for refactorization, then FGMRES, with KLU redo logic
        {
            std::cout << "DEBUG: System " << i << ": Using CuSolverRf refactorization and FGMRES." << std::endl;
            matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);

            status_refactor = Rf->refactorize(); // Attempt CuSolverRf refactorization
            std::cout << "CuSolverRf refactorization status: " << status_refactor << std::endl;

            // --- Hybrid Logic: Redo KLU if CuSolverRf crashed ---
            if (status_refactor != 0)
            {

                std::cout << "\n \t !!! ALERT !!! CuSolverRf has crashed; redoing KLU symbolic and numeric factorization. !!! ALERT !!! \n \n";

                // Redo KLU factorization
                KLU->setup(A);
                matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);
                status = KLU->analyze();
                std::cout << "KLU analysis status (redo): " << status << std::endl;
                status = KLU->factorize();
                std::cout << "KLU factorization status (redo): " << status << std::endl;
                status = KLU->solve(vec_rhs, vec_x); // Solve with new KLU factors
                std::cout << "KLU solve status (redo): " << status << std::endl;

                // Use the helper to print the summary after the KLU redo
                helper->resetSystem(A, vec_rhs, vec_x);
                helper->printShortSummary();

                // Re-setup CuSolverRf with the NEW KLU factors
                L_csc_klu = (ReSolve::matrix::Csc*) KLU->getLFactor();
                U_csc_klu = (ReSolve::matrix::Csc*) KLU->getUFactor();
                P_klu = KLU->getPOrdering();
                Q_klu = KLU->getQOrdering();

                // Check if KLU factors are valid before re-setting up CuSolverRf
                if (L_csc_klu == nullptr || U_csc_klu == nullptr) {
                    std::cerr << "ERROR: KLU factor pointers are null. Aborting CuSolverRf setup.\n";
                    goto cleanup;
                }

                Rf->setup(A, L_csc_klu, U_csc_klu, P_klu, Q_klu);
                Rf->refactorize(); // Re-refactorize CuSolverRf
            }

            status = Rf->solve(vec_rhs, vec_x); // Solve with CuSolverRf
            std::cout << "CuSolverRf solve status: " << status << std::endl;

            // Use the helper to print the summary after the refactorization solve
            helper->resetSystem(A, vec_rhs, vec_x);
            helper->printShortSummary();

            // Compute the residual: r = b - Ax
            vec_residual->copyDataFrom(vec_rhs, ReSolve::memory::DEVICE, ReSolve::memory::DEVICE);
            matrix_handler->matvec(A, vec_x, vec_residual, &ReSolve::constants::MINUS_ONE, &ReSolve::constants::ONE, ReSolve::memory::DEVICE);

            std::cout << "DEBUG: Solving error equation with FGMRES." << std::endl;

            FGMRES->resetMatrix(A); // Reset FGMRES with current matrix A
            FGMRES->solve(vec_residual, vec_error); // Refine solution with FGMRES
            std::cout << "FGMRES norm of error: " << sqrt(vector_handler->dot(vec_error, vec_error, ReSolve::memory::DEVICE)) << std::endl;

            // Print FGMRES summary using the helper function
            helper->printIrSummary(FGMRES);
            std::cout << "FGMRES Effective Stability: " << FGMRES->getEffectiveStability() << std::endl;

	    // Update the solution: x = x + e
            std::cout << "DEBUG: Updating solution vector." << std::endl;
            vector_handler->axpy(&ReSolve::constants::ONE, vec_error, vec_x, ReSolve::memory::DEVICE);

	    // Final residual calculation
            helper->resetSystem(A, vec_rhs, vec_x);
            std::cout << "DEBUG: Relative residual after error update: " << helper->getNormRelativeResidual() << std::endl;

	    // Setting vec_error to zero to get Relative residual norm
	    vec_error->setToZero(ReSolve::memory::HOST);
            vec_error->setToZero(ReSolve::memory::DEVICE);

        }
    } // End of for loop

cleanup: // Central cleanup label for error handling and end of program
    std::cout << "\n--- Cleaning up ReSolve objects ---" << std::endl;

    // Delete pointers only if they were successfully allocated (not nullptr)
    if (A) delete A; A = nullptr;
    if (KLU) delete KLU; KLU = nullptr;
    if (Rf) delete Rf; Rf = nullptr;
    if (FGMRES) delete FGMRES; FGMRES = nullptr;
    if (GS) delete GS; GS = nullptr;
    if (x_host_array) delete[] x_host_array; x_host_array = nullptr; // delete[] for C-style array
    if (rhs_host_array) delete[] rhs_host_array; rhs_host_array = nullptr; // delete[] for C-style array
    if (vec_x) delete vec_x; vec_x = nullptr;
    if (vec_rhs) delete vec_rhs; vec_rhs = nullptr;
    if (vector_handler) delete vector_handler; vector_handler = nullptr;
    if (matrix_handler) delete matrix_handler; matrix_handler = nullptr;
    if (workspace_CUDA) delete workspace_CUDA; workspace_CUDA = nullptr;
    if (helper) delete helper; helper = nullptr; // Cleanup the new helper object

    std::cout << "Cleanup complete. Program exiting." << std::endl;
    return 0; // Return 0 for success, or non-zero if goto was due to error
}
