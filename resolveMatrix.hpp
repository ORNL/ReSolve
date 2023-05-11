// Matrix utilities
// Mirroring memory approach 
#pragma once
#include <string>
namespace ReSolve {
  class resolveMatrix {

    public:
      //basic constructor
      resolveMatrix();
      resolveMatrix(int n, int m, int nnz);
      ~resolveMatrix();

      // accessors
      int getNumRows();
      int getNumColumns();
      int getNnz();

      int* getCsrRowPointers(std::string memspace);
      int* getCsrColIndices(std::string memspace);
      double* getCsrValues(std::string memspace);

      int* getCscColPointers(std::string memspace);
      int* getCscRowIndices(std::string memspace);
      double* getCscValues(std::string memspace);

      int* getCooRowIndices(std::string memspace);
      int* getCooColIndices(std::string memspace);
      double* getCooValues(std::string memspace);

      int setCsr(int* csr_p, int* csr_i, double* csr_x, std::string memspace);
      int setCsc(int* csc_p, int* csc_i, double* csc_x, std::string memspace);
      int setCoo(int* coo_rows, int* coo_cols, double* coo_vals, std::string memspace);

      //int setCudaWorkspace(resolveWorkspace* cudaWorkspace);
      //format conversion
      int convertCooToCsr(std::string memspace);
      int convertCsrToCsc(std::string memspace);

    private:
      //size
      int n;
      int m;
      int nnz;

      //memory space (cpu or gpu)
      std::string memspace;

      //host data
      // COO format:
      int* h_coo_rows;
      int* h_coo_cols;
      double* h_coo_vals;

      // CSR format:
      int* h_csr_p; //row starts
      int* h_csr_i; //column indices
      double* h_csr_x;//values 

      // CSC format:
      int* h_csc_p; //column starts
      int* h_csc_i; //row indices
      double* h_csc_x;//values 

      //device data

      /* note -- COO format not typically kept on the gpu anyways */

      //COO format:
      int* d_coo_rows;
      int* d_coo_cols;
      double* d_coo_vals;

      // CSR format:
      int* d_csr_p; //row starts
      int* d_csr_i; //column indices
      double* d_csr_x;//values 

      // CSC format:
      int* d_csc_p; //column starts
      int* d_csc_i; //row indices
      double* d_csc_x;//values 

      // if cuda functions are used, we need "workspace" that stores buffers and handles
      //resolveWorkspace* cuda_workspace;
  };
}
