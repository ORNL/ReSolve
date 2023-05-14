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
    std::fill_n(nnz_counts, n, 0);
    resolveInt* coo_rows = A->getCooRowIndices("cpu");
    resolveInt* coo_cols = A->getCooColIndices("cpu");
    resolveReal* coo_vals = A->getCooValues("cpu");

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
    printf("original A nnz: %d expanded nnz %d \n", nnz, nnz_unpacked);
    resolveInt* csr_ia = new resolveInt[n+1];
    std::fill_n(csr_ia, n + 1, 0);
    resolveInt* csr_ja = new resolveInt[nnz_unpacked];
    resolveReal* csr_a = new resolveReal[nnz_unpacked];
    resolveInt* nnz_shifts = new resolveInt[n];
    std::fill_n(nnz_shifts, n , 0);

    indexPlusValue* tmp = new indexPlusValue[nnz_unpacked]; 

    csr_ia[0] = 0;

    for (resolveInt i = 1; i < n + 1; ++i){
      csr_ia[i] = csr_ia[i - 1] + nnz_counts[i - 1];
    }

    int r, start;


    printf("\n\n");
    for (resolveInt i = 0; i < nnz; ++i){
      //which row
      r = coo_rows[i];
      start = csr_ia[r];

      if ((start + nnz_shifts[r]) > nnz_unpacked)
        printf("index out of bounds 1: start %d nnz_shifts[%d] = %d \n", start, r, nnz_shifts[r]);

      tmp[start + nnz_shifts[r]].setIdx(coo_cols[i]);
      tmp[start + nnz_shifts[r]].setValue(coo_vals[i]);

      nnz_shifts[r]++;

      if ((coo_rows[i] != coo_cols[i]) && (symmetric == 1))
      {

        r = coo_cols[i];
        start = csr_ia[r];

        if ((start + nnz_shifts[r]) > nnz_unpacked)
          printf("index out of bounds 2\n");
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
      A->updateCsr(csr_ia, csr_ja, csr_a, "cpu", "cpu");
    } else {
      if (memspace == "cuda"){      
      A->updateCsr(csr_ia, csr_ja, csr_a, "cpu", "cuda");
      } else {
      //display error
      }
    }
    delete [] nnz_counts;
    delete [] tmp;
    delete [] nnz_shifts;
    delete [] csr_ia;
    delete [] csr_ja;
    delete [] csr_a;

  }
}
