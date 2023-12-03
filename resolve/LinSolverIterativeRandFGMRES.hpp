#pragma once
#include "Common.hpp"
#include <resolve/matrix/Sparse.hpp>
#include <resolve/vector/Vector.hpp>
#include "LinSolver.hpp"
#include "GramSchmidt.hpp"
#include "RandSketchingManager.hpp"

namespace ReSolve 
{

  class LinSolverIterativeRandFGMRES : public LinSolverIterative
  {
    using vector_type = vector::Vector;

    public:
   
    enum SketchingMethod { cs = 0, // count sketch 
                           fwht = 1}; // fast Walsh-Hadamard transform
   
    LinSolverIterativeRandFGMRES(std::string memspace = "cuda");

    LinSolverIterativeRandFGMRES( MatrixHandler* matrix_handler,
                                  VectorHandler* vector_handler,
                                  SketchingMethod rand_method, 
                                  GramSchmidt*   gs,
                                  std::string memspace = "cuda");

    LinSolverIterativeRandFGMRES(index_type restart,
                                 real_type  tol,
                                 index_type maxit,
                                 index_type conv_cond,
                                 MatrixHandler* matrix_handler,
                                 VectorHandler* vector_handler, 
                                 SketchingMethod rand_method, 
                                 GramSchmidt*   gs,
                                 std::string memspace = "cuda");
    ~LinSolverIterativeRandFGMRES();

    int solve(vector_type* rhs, vector_type* x);
    int setup(matrix::Sparse* A);
    int resetMatrix(matrix::Sparse* new_A); 
    int setupPreconditioner(std::string name, LinSolverDirect* LU_solver);

    real_type getTol();
    index_type getMaxit();
    index_type getRestart();
    index_type getConvCond();
    bool getFlexible();
    std::string getRandSketchingMethod();
    index_type getKrand();

    void setTol(real_type new_tol);
    void setMaxit(index_type new_maxit);
    void setRestart(index_type new_restart);
    void setConvCond(index_type new_conv_cond);
    void setFlexible(bool new_flexible);
    void getRandSketchingMethod(std::string rand_method);

    private:
    //remember matrix handler and vector handler are inherited.

    std::string memspace_;

    real_type tol_;
    index_type maxit_;
    index_type restart_;
    std::string orth_option_;
    index_type conv_cond_;
    bool flexible_{1}; // if can be run as "normal" GMRES if needed, set flexible_ to 0. Default is 1 of course.

    vector_type* d_V_{nullptr};
    vector_type* d_Z_{nullptr};
    // for performing Gram-Schmidt
    vector_type* d_S_{nullptr};

    real_type* h_H_{nullptr};
    real_type* h_c_{nullptr};
    real_type* h_s_{nullptr};
    real_type* h_rs_{nullptr};
    real_type* d_aux_{nullptr};


    GramSchmidt* GS_;     
    void precV(vector_type* rhs, vector_type* x); //multiply the vector by preconditioner
    LinSolverDirect* LU_solver_;
    index_type n_;// for simplicity
    real_type one_over_k_{1.0};

    index_type k_rand_{0}; // size of sketch space, we need to know it so we can allocate S!
    MemoryHandler mem_; ///< Device memory manager object
    RandSketchingManager* rand_manager_;
    SketchingMethod rand_method_; 
  };
}
