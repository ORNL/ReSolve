#include "resolveMatrixHandler.hpp"

namespace ReSolve
{
  //helper class
  indexPlusValue::indexPlusValue()
  {
  }


  indexPlusValue::~indexPlusValue()
  {
  }

  void indexPlusValue::setIdx(resolveInt new_idx)
  {
    idx = new_idx;
  }

  void indexPlusValue::setValue(resolveReal new_value)
  {
    value = new_value;
  }

  resolveInt indexPlusValue::getIdx()
  {
    return idx;
  }

  resolveReal indexPlusValue::getValue()
  {
    return value;
  }

  //end of helper class


  resolveMatrixHandler::resolveMatrixHandler()
  {
  }

  resolveMatrixHandler::~resolveMatrixHandler()
  {
  }

  resolveMatrixHandler::resolveMatrixHandler(resolveMatrixHandlerWorkspace* new_workspace)
  {
    workspace = new_workspace;
  }

  void resolveMatrixHandler::coo2csr(resolveMatrix* A, std::string memspace)
  {
    //this happens on the CPU not on the GPU
    //but will return whatever memspace requested.

    //count nnzs first

    resolveInt nnz_unpacked = 0;
    resolveInt nnz = A->getNnz();
    resolveInt n = A->getNumRows();
    bool symmetric = A->symmetric();
    bool expanded = A->expanded();

    resolveInt* nnz_counts =  new resolveInt[n];

    resolveInt* coo_rows = A->getCsrRowPointers("cpu");
    resolveInt* coo_cols = A->getCsrColIndices("cpu");
    resolveReal* coo_vals = A->getCsrValues("cpu");

    //maybe check if they exist?
    
    for (resolveInt i = 0; i < nnz; ++i)
    {
      nnz_counts[coo_rows[i]]++;
      nnz_unpacked++;
      if ((coo_rows[i] != coo_cols[i])&& (symmetric) && (!expanded))
      {
        nnz_counts[coo_cols[i]]++;
        nnz_unpacked++;
      }
    }
    A->setExpanded(true);
    A->setNnzExpanded(nnz_unpacked);

    resolveInt* csr_ia = new resolveInt[n+1];
    resolveInt* csr_ja = new resolveInt[nnz_unpacked];
    resolveReal* csr_a = new resolveReal[nnz_unpacked];
    resolveInt* nnz_shifts = new resolveInt[n];

    indexPlusValue* tmp = new indexPlusValue[nnz_unpacked]; 

    csr_ia[0] = 0;

    for (resolveInt i = 1; i < n + 1; ++i){
      csr_ia[i] = csr_ia[i - 1] + nnz_counts[i - 1];
    }

    int r, start;

    for (resolveInt i = 0; i < nnz; ++i){
      //which row
      r = coo_rows[i];
      start = csr_ia[r];
      if ((start + nnz_shifts[r]) > nnz_unpacked)
        printf("index out of boubds\n");

      tmp[start + nnz_shifts[r]].setIdx(coo_cols[i]);
      tmp[start + nnz_shifts[r]].setValue(coo_vals[i]);

      nnz_shifts[r]++;

      if ((coo_rows[i] != coo_cols[i]) && (symmetric == 1))
      {

        r = coo_cols[i];
        start = csr_ia[r];

        if ((start + nnz_shifts[r]) > nnz_unpacked)
          printf("index out of boubds 2\n");
        tmp[start + nnz_shifts[r]].setIdx(coo_rows[i]);
        tmp[start + nnz_shifts[r]].setValue(coo_vals[i]);
        nnz_shifts[r]++;
      }
    }
    //now sort whatever is inside rows

    for (int i = 0; i < n; ++i)
    {

      //now sorting (and adding 1)
      int colStart = csr_ia[i];
      int colEnd = csr_ia[i + 1];
      int length = colEnd - colStart;

      std::sort(&tmp[colStart],&tmp[colStart] + length);
    }

    for (resolveInt i = 0; i < nnz_unpacked; ++i)
    {
      csr_ja[i] = tmp[i].getIdx();
      csr_a[i] = tmp[i].getValue();
    }

    if (memspace == "cpu"){
      A->setCsr(csr_ia, csr_ja, csr_a, nnz_unpacked, "cpu");
    } else {
      if (memspace == "cuda"){
        resolveInt* d_csr_ia;
        resolveInt* d_csr_ja;
        resolveReal* d_csr_a;
        cudaMalloc(&d_csr_ia, (n + 1)*sizeof(resolveInt)); 
        cudaMalloc(&d_csr_ja, (nnz_unpacked)*sizeof(resolveInt)); 
        cudaMalloc(&d_csr_a, (nnz_unpacked)*sizeof(resolveReal)); 

        cudaMemcpy(d_csr_ia, csr_ia, (n + 1) * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csr_ja, csr_ja, (nnz_unpacked) * sizeof(resolveInt), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csr_a, csr_a, (nnz_unpacked) * sizeof(resolveReal), cudaMemcpyHostToDevice);
      } else {
      //display error
      }
    }
    delete [] nnz_counts;
    delete [] tmp;
    delete [] nnz_shifts;

  }
}
