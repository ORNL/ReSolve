#pragma once

#include <cassert>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include <resolve/LinSolverIterative.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/matrix/Sparse.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/matrix/Csr.hpp>

// LAPACK Prototype for Least Squares (G*x = B)
extern "C" void dgels_(char* trans, int* m, int* n, int* nrhs,
                       double* a, int* lda, double* b, int* ldb,
                       double* work, int* lwork, int* info);

namespace ReSolve
{
  namespace examples
  {
    /**
     * @brief Prints linear system info.
     *
     * @param name - pathname of the system matrix
     * @param A    - pointer to the system matrix
     */
    void printSystemInfo(const std::string& matrix_pathname, matrix::Sparse* A)
    {
      std::cout << std::endl;
      std::cout << "========================================================================================================================\n";
      std::cout << "Reading: " << matrix_pathname << std::endl;
      std::cout << "========================================================================================================================\n";
      std::cout << std::endl;

      std::cout << "Finished reading the matrix and rhs, size: " << A->getNumRows() << " x " << A->getNumColumns()
                << ", nnz: " << A->getNnz()
                << ", symmetric? " << A->symmetric()
                << ", Expanded? " << A->expanded() << std::endl;
    }

    /**
     * @brief Test helper class template
     *
     * This is header-only implementation of several utility functions used by
     * multiple functionality tests, such as error norm calculations. To use,
     * simply include this header in the test.
     *
     * @tparam workspace_type
     */
    template <class workspace_type>
    class ExampleHelper
    {
    public:
      /**
       * @brief Default constructor
       *
       * Initializes matrix and vector handlers.
       *
       * @param[in,out] workspace - workspace for matrix and vector handlers
       *
       * @pre Workspace handles are initialized
       *
       * @post Handlers are instantiated.
       * allocated
       */
      ExampleHelper(workspace_type& workspace)
        : mh_(&workspace),
          vh_(&workspace)
      {
        memspace_ = ReSolve::memory::DEVICE;
        if (mh_.getIsCudaEnabled())
        {
          hardware_backend_ = "CUDA";
        }
        else if (mh_.getIsHipEnabled())
        {
          hardware_backend_ = "HIP";
        }
        else
        {
          hardware_backend_ = "CPU";
          memspace_         = ReSolve::memory::HOST;
        }
      }

      /**
       * @brief Destroy the ExampleHelper object
       *
       * @post Vectors res_ and x_true_ are deleted.
       *
       */
      ~ExampleHelper()
      {
        if (res_)
        {
          delete res_;
          res_ = nullptr;
        }
        if (x_true_)
        {
          delete x_true_;
          x_true_ = nullptr;
        }
      }

      /// Returns the configured hardware backend
      std::string getHardwareBackend() const
      {
        return hardware_backend_;
      }

      /// Returns the configured memory space
      ReSolve::memory::MemorySpace getMemspace() const
      {
        return memspace_;
      }

      /**
       * @brief Set the new linear system together with its computed solution
       * and compute solution error and residual norms.
       *
       * This will set the new system A*x = r and compute related error norms.
       *
       * @param A[in] - Linear system matrix
       * @param r[in] - Linear system right-hand side
       * @param x[in] - Computed solution of the linear system
       */
      void setSystem(ReSolve::matrix::Sparse* A,
                     ReSolve::vector::Vector* r,
                     ReSolve::vector::Vector* x)
      {
        assert((res_ == nullptr) && (x_true_ == nullptr));
        A_   = A;
        r_   = r;
        x_   = x;
        res_ = new ReSolve::vector::Vector(A->getNumRows());
        computeNorms();
      }

      /**
       * @brief Set the new linear system together with its computed solution
       * and compute solution error and residual norms.
       *
       * This is to be used after values in A and r are updated.
       *
       * @todo This method probably does not need any input parameters.
       *
       * @param A[in] - Linear system matrix
       * @param r[in] - Linear system right-hand side
       * @param x[in] - Computed solution of the linear system
       */
      void resetSystem(ReSolve::matrix::Sparse* A,
                       ReSolve::vector::Vector* r,
                       ReSolve::vector::Vector* x)
      {
        A_ = A;
        r_ = r;
        x_ = x;
        if (res_ == nullptr)
        {
          res_ = new ReSolve::vector::Vector(A->getNumRows());
        }

        computeNorms();
      }

      /// Return L2 norm of the linear system residual.
      ReSolve::real_type getNormResidual()
      {
        return norm_res_;
      }

      /// Return relative residual norm.
      ReSolve::real_type getNormRelativeResidual()
      {
        return norm_res_ / norm_rhs_;
      }

      /// Minimalistic summary
      void printShortSummary()
      {
        std::cout << "\t2-Norm of the residual: "
                  << std::scientific << std::setprecision(16)
                  << getNormRelativeResidual() << "\n";
      }

      /// Summary of direct solve
      void printSummary()
      {
        std::cout << "\t 2-Norm of the residual (before IR): "
                  << std::scientific << std::setprecision(16)
                  << getNormRelativeResidual() << "\n";

        std::cout << std::scientific << std::setprecision(16)
                  << "\t Matrix inf  norm: " << inf_norm_A_ << "\n"
                  << "\t Residual inf norm: " << inf_norm_res_ << "\n"
                  << "\t Solution inf norm: " << inf_norm_x_ << "\n"
                  << "\t Norm of scaled residuals: " << nsr_norm_ << "\n";
      }

      /// Summary of error norms for an iterative refinement test.
      void printIrSummary(ReSolve::LinSolverIterative* ls)
      {
        std::cout << "FGMRES: init nrm: "
                  << std::scientific << std::setprecision(16)
                  << ls->getInitResidualNorm() / norm_rhs_
                  << " final nrm: "
                  << ls->getFinalResidualNorm() / norm_rhs_
                  << " iter: " << ls->getNumIter() << "\n";
      }

      /// Summary of error norms for an iterative solver test.
      void printIterativeSolverSummary(ReSolve::LinSolverIterative* ls)
      {
        std::cout << std::setprecision(16) << std::scientific;
        std::cout << "\t Initial residual norm          ||b-A*x||       : " << ls->getInitResidualNorm() << "\n";
        std::cout << "\t Initial relative residual norm ||b-A*x||/||b|| : " << ls->getInitResidualNorm() / norm_rhs_ << "\n";
        std::cout << "\t Final residual norm            ||b-A*x||       : " << ls->getFinalResidualNorm() << "\n";
        std::cout << "\t Final relative residual norm   ||b-A*x||/||b|| : " << ls->getFinalResidualNorm() / norm_rhs_ << "\n";
        std::cout << "\t Number of iterations                           : " << ls->getNumIter() << "\n";
      }

      /// Check the relative residual norm against `tolerance`.
      int checkResult(ReSolve::real_type tolerance)
      {
        int                error_sum = 0;
        ReSolve::real_type norm      = norm_res_ / norm_rhs_;

        if (!std::isfinite(norm))
        {
          std::cout << "Result is not a finite number!\n";
          error_sum++;
        }
        if (norm > tolerance)
        {
          std::cout << "Result inaccurate!\n";
          error_sum++;
        }

        return error_sum;
      }

      /**
       * @breif Enforce row-equilibration for the system A and rhs
       * Calculates the infinity norm of each row of the matrix
       * assembles them into a ReSolve::Vector::vector
       * Scales each row using matrix_handler class
       * @param in - pointer to A and RHS
       * @param out - scaled version of A and b
       */

       void rowEquilibration(ReSolve::matrix::Csr* A, ReSolve::vector::Vector* rhs)
       {
	 using namespace ReSolve;
	 using vector_type = ReSolve::vector::Vector;
	 using index_type = ReSolve::index_type;
         using real_type = ReSolve::real_type;

         vector_type* diag = nullptr; // HOST vector for containing the inf norms of A_
         diag = new vector_type(A->getNumRows());
	 diag->allocate(ReSolve::memory::HOST);
         diag->setToZero(ReSolve::memory::HOST);

	 // Extracting row and val data from A_
         index_type* rowData = A->getRowData(ReSolve::memory::HOST);
         real_type* values = A->getValues(ReSolve::memory::HOST);
	 real_type* diagData = new real_type[A->getNumRows()];

	 for (index_type i = 0; i < A->getNumRows(); ++i) {
	     real_type rowMax = 0.0;

	     // Compute max absolute value in row i
	     for (index_type j = rowData[i]; j < rowData[i + 1]; ++j) {
		 real_type absVal = std::abs(values[j]);
		 if (absVal > rowMax) rowMax = absVal;
	     }

	     // Store the multiplier
	     diagData[i] = (rowMax > 0.0) ? (1.0 / rowMax) : 1.0;
	 }

	    // Wrap raw data into diag vector and sync to DEVICE
	    diag->copyDataFrom(diagData, ReSolve::memory::HOST, ReSolve::memory::HOST);

	    // Scaling A and RHS
	    std::cout << "Scaling the matrix A" << std::endl;
	    mh_.leftScale(diag, A, ReSolve::memory::HOST);
	    std::cout << "Scaling the RHS" << std::endl;
	    vh_.scale(diag, rhs, ReSolve::memory::HOST);

	    if (diag) { delete diag; diag = nullptr; }
       }

      /**
       * @brief Reads a Triplet text file and populates a ReSolve CSR matrix.
       * @param in - filename, number of rows and number of columns
       * @param out - pattern of SAM PP
       */
      void loadSAMPattern(const std::string& filename, int numRows, int numCols, ReSolve::matrix::Csr*& outputCSR)
      {
	using namespace ReSolve;
	using index_type = ReSolve::index_type;
	using real_type = ReSolve::real_type;

	std::ifstream file(filename);
	if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    	}

	std::vector<index_type> rows, cols;
    	std::vector<real_type> vals;
    	int r, c;
    	double v;

	// 1. Read the triplet data from the file
	while (file >> r >> c >> v) {
           rows.push_back(r);
           cols.push_back(c);
           vals.push_back(v);
    	}
    	file.close();

	index_type nnz = rows.size();

	// 2. Build the CSR row_ptr (Counting Sort Phase 1)
	std::vector<index_type> row_ptr(numRows + 1, 0);
	for (index_type i = 0; i < nnz; ++i) {
          row_ptr[rows[i] + 1]++;
    	}

	// Transform counts into offsets (Prefix Sum)
    	for (int i = 0; i < numRows; ++i) {
          row_ptr[i + 1] += row_ptr[i];
    	}

	// 3. Populate Column Indices and Values (Counting Sort Phase 2)
	std::vector<index_type> col_indices(nnz);
	std::vector<real_type> values(nnz);
	std::vector<int> current_row_offset = row_ptr; // Copy to track progress per row

	for (index_type i = 0; i < nnz; ++i) {
          int row = rows[i];
          int dest = current_row_offset[row]++;
          col_indices[dest] = cols[i];
          values[dest] = vals[i];
        }

	// 4. Sorting column indices within each row
	for (int i = 0; i < numRows; ++i) {
          int start = row_ptr[i];
          int end = row_ptr[i + 1];

          // We need to sort both col_indices and values based on col_indices.
          // For a Boolean pattern (all values = 1.0), we can just sort the indices.
          std::sort(col_indices.begin() + start, col_indices.begin() + end);
        }

	// 5. Hand over the data to the ReSolve object
	outputCSR = new ReSolve::matrix::Csr(numRows, numCols, nnz);
	outputCSR->allocateMatrixData(ReSolve::memory::HOST);

	outputCSR->copyDataFrom(
            row_ptr.data(),
            col_indices.data(),
            values.data(),
            ReSolve::memory::HOST,
            ReSolve::memory::HOST
    	);

	std::cout << "Successfully loaded SAM Pattern: " << outputCSR->getNnz() << " non-zeros." << std::endl;

      }

      /**
       * @brief SpGEMM for SAM Computation
       *
       * Calculate the sparsity pattern of A*P, where A and P are user provided.
       * @param in - pointer to A and P
       * @param out - pointer to A*P Boolean matrix
       */
      void spGEMM(ReSolve::matrix::Csr* A, ReSolve::matrix::Csr* P, ReSolve::matrix::Csr*& PP3)
      {
        using namespace ReSolve;
	using index_type = ReSolve::index_type;
	using real_type = ReSolve::real_type;

	index_type num_rows_A = A->getNumRows();
	index_type num_cols_M = P->getNumColumns(); // Use P's columns, not A's rows

	// 1. Symbolic Pass
	std::vector<index_type> row_ptr_C(num_rows_A + 1, 0);
	std::vector<index_type> col_ind_C; // This stores our actual column results
	std::vector<int> marker(num_cols_M, -1);

	// Extracting row pointers from A and P
	index_type* row_ptr_A = A->getRowData(ReSolve::memory::HOST);
	index_type* col_ind_A = A->getColData(ReSolve::memory::HOST);

	index_type* row_ptr_M = P->getRowData(ReSolve::memory::HOST);
	index_type* col_ind_M = P->getColData(ReSolve::memory::HOST);

	for (index_type i = 0; i < num_rows_A; ++i) {
      	    row_ptr_C[i] = (index_type)col_ind_C.size();

    	   // For each non-zero A(i, j)
    	   for (index_type jj = row_ptr_A[i]; jj < row_ptr_A[i+1]; ++jj) {
        	index_type j = col_ind_A[jj];

        	// For each non-zero M(j, k)
        	for (index_type kk = row_ptr_M[j]; kk < row_ptr_M[j+1]; ++kk) {
            	     index_type k = col_ind_M[kk];

            	    // If column k hasn't been added to row i yet
            	    if (marker[k] != (int)i) {
                	marker[k] = (int)i;
                	col_ind_C.push_back(k);
            	    }
        	}
    	   }
    	   // Sort the indices for the current row only
    	   std::sort(col_ind_C.begin() + row_ptr_C[i], col_ind_C.end());
	}
	row_ptr_C[num_rows_A] = (index_type)col_ind_C.size();

	// 2. Populate the ReSolve CSR pointer
	index_type total_nnz = row_ptr_C[num_rows_A];
	std::vector<real_type> ones(total_nnz, 1.0);

	// Allocation
	PP3 = new ReSolve::matrix::Csr(num_rows_A, num_cols_M, total_nnz);
	PP3->allocateMatrixData(ReSolve::memory::HOST);

	// Copying data
	PP3->copyDataFrom(
    	   row_ptr_C.data(),
    	   col_ind_C.data(),
    	   ones.data(),
    	   ReSolve::memory::HOST,
    	   ReSolve::memory::HOST
	);

        PP3->allocateMatrixData(ReSolve::memory::DEVICE);
        PP3->syncData(ReSolve::memory::DEVICE);

        std::cout <<"spGEMM is successful " << PP3->getNnz() << " non-zeros." << std::endl;
      }

      /**
       * @brief Pre-processing of SAM to determine the buffer for Least-Sq solve
       *
       * @param in - pointers to PP and PP3.
       * @param out - maxRow and maxCol
       *
       */
      void preSAM(ReSolve::matrix::Csr* PP, ReSolve::matrix::Csr* PP3, int& maxRow, int& maxCol)
       {
         using namespace ReSolve;
	 using index_type = ReSolve::index_type;

	 int num_cols = PP->getNumRows();

	 // Allocate counters for each column
	 std::vector<int> nnz_M(num_cols, 0);
	 std::vector<int> nnz_LS(num_cols, 0);

	 // Count non-zeros in P_SAM (these are the 'columns' of M)
	 index_type nnz_P = PP->getNnz();
	 index_type* col_ind_P = PP->getColData(ReSolve::memory::HOST);

	 for (index_type i = 0; i < nnz_P; ++i) {
            nnz_M[col_ind_P[i]]++;
    	 }

	 // Count non-zeros in PP3
	 index_type nnz_PP3 = PP3->getNnz();
	 index_type* col_ind_PP3 = PP3->getColData(ReSolve::memory::HOST);

	 for (index_type i = 0; i < nnz_PP3; ++i) {
	    nnz_LS[col_ind_PP3[i]]++;
	 }

	 // Find the maximums
	 maxCol = *std::max_element(nnz_M.begin(), nnz_M.end());
	 maxRow = *std::max_element(nnz_LS.begin(), nnz_LS.end());

	 std::cout << "Workspace Dims - Max Row (PP3): " << maxRow
              << ", Max Col (PP): " << maxCol << std::endl;
       }

      /**
       * @brief Creates an inverted index (CSC-like mapping) for a CSR matrix.
       * This allows us to simulate MATLAB's find(PP(:, k)) efficiently.
       */
      std::vector<std::vector<int>> buildColToRowMap(ReSolve::matrix::Csr* mat)
      {
	  int numCols = mat->getNumRows();
	  int numRows = numCols;
	  std::vector<std::vector<int>> colToRows(numCols);

	  // Create a vector of vectors: results[col_index] = {row1, row2, ...}
	  int* rowPtr = mat->getRowData(ReSolve::memory::HOST);
	  int* colInd = mat->getColData(ReSolve::memory::HOST);

	  // Iterate through every non-zero entry in the CSR matrix
	  for (int i = 0; i < numRows; ++i) {
	      for (int jj = rowPtr[i]; jj < rowPtr[i + 1]; ++jj) {
		  int j = colInd[jj];
		  // Add the current row index (i) to the list for column (j)
		  colToRows[j].push_back(i);
	      }
	  }

	  return colToRows;
      }

      /**
       * @brief Helper to find a value in a CSR row manually.
       * O(log(row_nnz)) per call.
       */
      double findValueInRow(int row, int target_col,
                      const index_type* row_ptr,
                      const index_type* col_ind,
                      const real_type* values) {

	index_type start = row_ptr[row];
	index_type end = row_ptr[row + 1];

	// Standard binary search for the column index
	auto it = std::lower_bound(col_ind + start, col_ind + end, target_col);

	if (it != (col_ind + end) && *it == target_col) {
	    index_type offset = std::distance(col_ind, it);
	    return (double)values[offset];
	}

	return 0.0; // structural zero

      }

      /**
       * @brief Computing SAM column by column
       *
       * @param in - pointers to Ak, PP, A1, maxRow, maxCol
       * @param out - map
       *
       */
      void computeSAM(ReSolve::matrix::Csr* A, ReSolve::matrix::Csr* A1, ReSolve::matrix::Csr* P, ReSolve::matrix::Csr* PP3, int maxRow, int maxCol)
       {
	 using namespace ReSolve;
	 using index_type = ReSolve::index_type;
	 using real_type = ReSolve::real_type;

	 int n = A->getNumRows();

	 // Using inverted indices logic to simulate find(PP(:,k))
	 auto rowsInM = buildColToRowMap(P);
	 auto rowsInLs = buildColToRowMap(PP3);

	 // Pre-allocating workspaces
	 // G must be large enough for the biggest possible local system
	 std::vector<double> G_dense(maxRow * maxCol);
	 std::vector<double> B_vec(maxRow);

	 // Result buffers for the final SAM matrix (Triplets)
	 std::vector<index_type> rowM, colM;
	 std::vector<real_type> valM;

	 // Get raw pointers for A and A1
	 const index_type* row_ptr_A = A->getRowData(ReSolve::memory::HOST);
	 const index_type* col_ind_A = A->getColData(ReSolve::memory::HOST);
	 const real_type* values_A  = A->getValues(ReSolve::memory::HOST);

	 const index_type* row_ptr_A1 = A1->getRowData(ReSolve::memory::HOST);
	 const index_type* col_ind_A1 = A1->getColData(ReSolve::memory::HOST);
	 const real_type* values_A1  = A1->getValues(ReSolve::memory::HOST);

	 // Workspace for LAPACK (optimal size query)
	 int lwork = -1;
	 double wkopt;
	 int nrhs = 1;
	 int info;
	 char trans = 'N';

	 // Column-by-column computation
	 for (int k = 0; k < n; ++k){
	     const auto& unknowns = rowsInM[k];   // col = find(PP(:,k))
	     const auto& equations = rowsInLs[k]; // row = find(PP3(:,k))

	     int n_u = (int)unknowns.size();
	     int n_e = (int)equations.size();

	     if (n_u == 0 || n_e == 0) continue;

	     // Numerical Extraction: Fill G and B
	     for (int i = 0; i < n_e; ++i){
		int g_row = equations[i];

		// Fill vector B: B[i] = A(global_row, k)
		B_vec[i] = findValueInRow(g_row, k, row_ptr_A1, col_ind_A1, values_A1);

		// Fill G: G(i, j) = A(global_row, unknowns[j])
	        for (int j = 0; j < n_u; ++j){
		    G_dense[i + j * n_e] = findValueInRow(g_row, unknowns[j], row_ptr_A, col_ind_A, values_A);
		}
	     }

	     // Solve least squares: G * x = B
	     dgels_(&trans, &n_e, &n_u, &nrhs, G_dense.data(), &n_e,
                    B_vec.data(), &n_e, &wkopt, &lwork, &info);

	     int current_lwork = (int)wkopt;
	     std::vector<double> work(current_lwork);

	     // Actual Solve
	     dgels_(&trans, &n_e, &n_u, &nrhs, G_dense.data(), &n_e,
                    B_vec.data(), &n_e, work.data(), &current_lwork, &info);

	     if (info != 0) {
             	std::cerr << "Warning: LAPACK dgels failed at column " << k << " with info " << info << std::endl;
             }

	     // Storing results (solution x is returned in the first n_u elements of B_vec)
	     for (int i = 0; i < n_u; ++i){
		 rowM.push_back(unknowns[i]); // Original row index of the pattern
		 colM.push_back(k);           // Current column
		 valM.push_back(B_vec[i]);    // Calculated value
	     }
	 }
	 // Rebuilding the SAM matrix
	 if (!valM.empty()) {
	    // Find the maximum absolute value (Magnitude)
	    auto max_it = std::max_element(valM.begin(), valM.end(),
        	 [](double a, double b) {
            	      return std::abs(a) < std::abs(b);
        	 });

	    double maxVal = std::abs(*max_it);

	    std::cout << "------------------------------------------" << std::endl;
	    std::cout << "SAM Numerical extraction complete." << std::endl;
	    std::cout << "Total entries: " << valM.size() << std::endl;
	    std::cout << "Largest entry (abs): " << maxVal << std::endl;
	    std::cout << "------------------------------------------" << std::endl;
	    }else{
	        std::cout << "Warning: valM is empty. No values were extracted." << std::endl;
	    }
       }


      /**
       * @brief Verify the computation of the norm of scaled residuals.
       *
       * The norm value is provided as the input. This function computes
       * the norm of scaled residuals for the system that has been set
       * by the constructor or (re)setSystem functions.
       *
       * @param nsr_system - norm of scaled residuals value to be verified
       * @return int - 0 if the result is correct, error code otherwise
       */
      int checkNormOfScaledResiduals(ReSolve::real_type nsr_system)
      {
        using namespace ReSolve;
        int error_sum = 0;

        // Compute residual norm to get updated vector res_
        res_->copyDataFrom(r_, memspace_, memspace_);
        norm_res_ = computeResidualNorm(*A_, *x_, *res_, memspace_);

        // Compute norm of scaled residuals
        real_type inf_norm_A = 0.0;
        mh_.matrixInfNorm(A_, &inf_norm_A, memspace_);
        real_type inf_norm_x   = vh_.infNorm(x_, memspace_);
        real_type inf_norm_res = vh_.infNorm(res_, memspace_);
        real_type nsr_norm     = inf_norm_res / (inf_norm_A * inf_norm_x);
        real_type error        = std::abs(nsr_system - nsr_norm) / nsr_norm;

        // Test norm of scaled residuals method in SystemSolver
        if (error > 10.0 * std::numeric_limits<real_type>::epsilon())
        {
          std::cout << "Norm of scaled residuals computation failed:\n";
          std::cout << std::scientific << std::setprecision(16)
                    << "\tMatrix inf  norm                 : " << inf_norm_A << "\n"
                    << "\tResidual inf norm                : " << inf_norm_res << "\n"
                    << "\tSolution inf norm                : " << inf_norm_x << "\n"
                    << "\tNorm of scaled residuals         : " << nsr_norm << "\n"
                    << "\tNorm of scaled residuals (system): " << nsr_system << "\n\n";
        }
        return error_sum;
      }

      /**
       * @brief Verify the computation of the relative residual norm.
       *
       * The norm value is provided as the input. This function computes
       * the relative residual norm for the system that has been set
       * by the constructor or (re)setSystem functions.
       *
       * @param rrn_system - relative residual norm value to be verified
       * @return int - 0 if the result is correct, error code otherwise
       */
      int checkRelativeResidualNorm(ReSolve::real_type rrn_system)
      {
        using namespace ReSolve;
        int error_sum = 0;

        // Compute residual norm
        res_->copyDataFrom(r_, memspace_, memspace_);
        norm_res_ = computeResidualNorm(*A_, *x_, *res_, memspace_);

        real_type error = std::abs(norm_rhs_ * rrn_system - norm_res_) / norm_res_;
        if (error > 10.0 * std::numeric_limits<real_type>::epsilon())
        {
          std::cout << "Relative residual norm computation failed:\n";
          std::cout << std::scientific << std::setprecision(16)
                    << "\tTest value            : " << norm_res_ / norm_rhs_ << "\n"
                    << "\tSystemSolver computed : " << rrn_system << "\n\n";
          error_sum++;
        }
        return error_sum;
      }

      /**
       * @brief Verify the computation of the residual norm.
       *
       * The norm value is provided as the input. This function computes
       * the residual norm for the system that has been set by the constructor
       * or (re)setSystem functions.
       *
       * @param rrn_system - residual norm value to be verified
       * @return int - 0 if the result is correct, error code otherwise
       */
      int checkResidualNorm(ReSolve::real_type rn_system)
      {
        using namespace ReSolve;
        int error_sum = 0;

        // Compute residual norm
        res_->copyDataFrom(r_, memspace_, memspace_);
        norm_res_ = computeResidualNorm(*A_, *x_, *res_, memspace_);

        real_type error = std::abs(rn_system - norm_res_) / norm_res_;
        if (error > 10.0 * std::numeric_limits<real_type>::epsilon())
        {
          std::cout << "Residual norm computation failed:\n";
          std::cout << std::scientific << std::setprecision(16)
                    << "\tTest value            : " << norm_res_ << "\n"
                    << "\tSystemSolver computed : " << rn_system << "\n\n";
          error_sum++;
        }
        return error_sum;
      }

    private:
      /// Compute error norms.
      void computeNorms()
      {
        // Compute rhs and residual norms
        res_->copyDataFrom(r_, memspace_, memspace_);
        norm_rhs_ = norm2(*r_, memspace_);
        norm_res_ = computeResidualNorm(*A_, *x_, *res_, memspace_);

        // Compute norm of scaled residuals
        mh_.matrixInfNorm(A_, &inf_norm_A_, memspace_);
        inf_norm_x_   = vh_.infNorm(x_, memspace_);
        inf_norm_res_ = vh_.infNorm(res_, memspace_);
        nsr_norm_     = inf_norm_res_ / (inf_norm_A_ * inf_norm_x_);
      }

      /**
       * @brief Computes residual norm = || A * x - r ||_2
       *
       * @param[in]     A - system matrix
       * @param[in]     x - computed solution of the system
       * @param[in,out] r - system right-hand side, residual vector
       * @param[in]     memspace memory space where to computate the norm
       * @return ReSolve::real_type
       *
       * @post r is overwritten with residual values
       */
      ReSolve::real_type computeResidualNorm(ReSolve::matrix::Sparse&     A,
                                             ReSolve::vector::Vector&     x,
                                             ReSolve::vector::Vector&     r,
                                             ReSolve::memory::MemorySpace memspace)
      {
        using namespace ReSolve::constants;
        mh_.matvec(&A, &x, &r, &ONE, &MINUS_ONE, memspace); // r := A * x - r
        return norm2(r, memspace);
      }

      /// Compute L2 norm of vector `r` in memory space `memspace`.
      ReSolve::real_type norm2(ReSolve::vector::Vector&     r,
                               ReSolve::memory::MemorySpace memspace)
      {
        return std::sqrt(vh_.dot(&r, &r, memspace));
      }

    private:
      ReSolve::matrix::Sparse* A_; ///< pointer to system matrix
      ReSolve::vector::Vector* r_; ///< pointer to system right-hand side
      ReSolve::vector::Vector* x_; ///< pointer to the computed solution

      ReSolve::MatrixHandler mh_; ///< matrix handler instance
      ReSolve::VectorHandler vh_; ///< vector handler instance

      ReSolve::vector::Vector* res_{nullptr};    ///< pointer to residual vector
      ReSolve::vector::Vector* x_true_{nullptr}; ///< pointer to solution error vector

      ReSolve::real_type norm_rhs_{0.0}; ///< right-hand side vector norm
      ReSolve::real_type norm_res_{0.0}; ///< residual vector norm

      real_type inf_norm_A_{0.0};   ///< infinity norm of matrix A
      real_type inf_norm_x_{0.0};   ///< infinity norm of solution x
      real_type inf_norm_res_{0.0}; ///< infinity norm of res = A*x - r
      real_type nsr_norm_{0.0};     ///< norm of scaled residuals

      ReSolve::memory::MemorySpace memspace_{ReSolve::memory::HOST};
      std::string                  hardware_backend_{"NONE"};
    };

  } // namespace examples
} // namespace ReSolve
