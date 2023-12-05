#include <cstring>  // <-- includes memcpy
#include <algorithm>
#include <cassert>

#include "Csr.hpp"
#include "Coo.hpp"
#include <resolve/utilities/misc/IndexValuePair.hpp>
#include <resolve/utilities/logger/Logger.hpp>

namespace ReSolve 
{
  using out = io::Logger;

  matrix::Csr::Csr()
  {
  }

  matrix::Csr::Csr(index_type n, index_type m, index_type nnz) : Sparse(n, m, nnz)
  {
  }
  
  matrix::Csr::Csr(index_type n, 
                   index_type m, 
                   index_type nnz,
                   bool symmetric,
                   bool expanded) : Sparse(n, m, nnz, symmetric, expanded)
  {
  }

  matrix::Csr::Csr(matrix::Coo* A_coo, memory::MemorySpace memspace)
    : Sparse(A_coo->getNumRows(),
             A_coo->getNumColumns(),
             A_coo->getNnz(),
             A_coo->symmetric(),
             A_coo->expanded())
  {
    coo2csr(A_coo, memspace);
  }

  matrix::Csr::~Csr()
  {
  }

  index_type* matrix::Csr::getRowData(memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    copyData(memspace);
    switch (memspace) {
      case HOST:
        return this->h_row_data_;
      case DEVICE:
        return this->d_row_data_;
      default:
        return nullptr;
    }
  }

  index_type* matrix::Csr::getColData(memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    copyData(memspace);
    switch (memspace) {
      case HOST:
        return this->h_col_data_;
      case DEVICE:
        return this->d_col_data_;
      default:
        return nullptr;
    }
  }

  real_type* matrix::Csr::getValues(memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    copyData(memspace);
    switch (memspace) {
      case HOST:
        return this->h_val_data_;
      case DEVICE:
        return this->d_val_data_;
      default:
        return nullptr;
    }
  }

  int matrix::Csr::updateData(index_type* row_data, index_type* col_data, real_type* val_data, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut)
  {
    //four cases (for now)
    index_type nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    setNotUpdated();
    int control = -1;
    if ((memspaceIn == memory::HOST)     && (memspaceOut == memory::HOST))    { control = 0;}
    if ((memspaceIn == memory::HOST)     && ((memspaceOut == memory::DEVICE))){ control = 1;}
    if (((memspaceIn == memory::DEVICE)) && (memspaceOut == memory::HOST))    { control = 2;}
    if (((memspaceIn == memory::DEVICE)) && ((memspaceOut == memory::DEVICE))){ control = 3;}

    if (memspaceOut == memory::HOST) {
      //check if cpu data allocated
      if (h_row_data_ == nullptr) {
        this->h_row_data_ = new index_type[n_ + 1];
      }
      if (h_col_data_ == nullptr) {
        this->h_col_data_ = new index_type[nnz_current];
      } 
      if (h_val_data_ == nullptr) {
        this->h_val_data_ = new real_type[nnz_current];
      }
    }

    if (memspaceOut == memory::DEVICE) {
      //check if cuda data allocated
      if (d_row_data_ == nullptr) {
        mem_.allocateArrayOnDevice(&d_row_data_, n_ + 1); 
      }
      if (d_col_data_ == nullptr) {
        mem_.allocateArrayOnDevice(&d_col_data_, nnz_current);
      }
      if (d_val_data_ == nullptr) {
        mem_.allocateArrayOnDevice(&d_val_data_, nnz_current); 
      }
    }


    //copy	
    switch(control)  {
      case 0: //cpu->cpu
        mem_.copyArrayHostToHost(h_row_data_, row_data,      n_ + 1);
        mem_.copyArrayHostToHost(h_col_data_, col_data, nnz_current);
        mem_.copyArrayHostToHost(h_val_data_, val_data, nnz_current);
        h_data_updated_ = true;
        owns_cpu_data_ = true;
        owns_cpu_vals_ = true;
        break;
      case 2://gpu->cpu
        mem_.copyArrayDeviceToHost(h_row_data_, row_data,      n_ + 1);
        mem_.copyArrayDeviceToHost(h_col_data_, col_data, nnz_current);
        mem_.copyArrayDeviceToHost(h_val_data_, val_data, nnz_current);
        h_data_updated_ = true;
        owns_cpu_data_ = true;
        owns_cpu_vals_ = true;
        break;
      case 1://cpu->gpu
        mem_.copyArrayHostToDevice(d_row_data_, row_data,      n_ + 1);
        mem_.copyArrayHostToDevice(d_col_data_, col_data, nnz_current);
        mem_.copyArrayHostToDevice(d_val_data_, val_data, nnz_current);
        d_data_updated_ = true;
        owns_gpu_data_ = true;
        owns_gpu_vals_ = true;
        break;
      case 3://gpu->gpu
        mem_.copyArrayDeviceToDevice(d_row_data_, row_data,      n_ + 1);
        mem_.copyArrayDeviceToDevice(d_col_data_, col_data, nnz_current);
        mem_.copyArrayDeviceToDevice(d_val_data_, val_data, nnz_current);
        d_data_updated_ = true;
        owns_gpu_data_ = true;
        owns_gpu_vals_ = true;
        break;
      default:
        return -1;
    }
    return 0;
  } 

  int matrix::Csr::updateData(index_type* row_data, index_type* col_data, real_type* val_data, index_type new_nnz, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut)
  {
    this->destroyMatrixData(memspaceOut);
    this->nnz_ = new_nnz;
    int i = this->updateData(row_data, col_data, val_data, memspaceIn, memspaceOut);
    return i;
  } 

  int matrix::Csr::allocateMatrixData(memory::MemorySpace memspace)
  {
    index_type nnz_current = nnz_;
    if (is_expanded_) {nnz_current = nnz_expanded_;}
    destroyMatrixData(memspace);//just in case

    if (memspace == memory::HOST) {
      this->h_row_data_ = new index_type[n_ + 1];
      std::fill(h_row_data_, h_row_data_ + n_ + 1, 0);  
      this->h_col_data_ = new index_type[nnz_current];
      std::fill(h_col_data_, h_col_data_ + nnz_current, 0);  
      this->h_val_data_ = new real_type[nnz_current];
      std::fill(h_val_data_, h_val_data_ + nnz_current, 0.0);  
      owns_cpu_data_ = true;
      owns_cpu_vals_ = true;
      return 0;   
    }

    if (memspace == memory::DEVICE) {
      mem_.allocateArrayOnDevice(&d_row_data_,      n_ + 1); 
      mem_.allocateArrayOnDevice(&d_col_data_, nnz_current); 
      mem_.allocateArrayOnDevice(&d_val_data_, nnz_current); 
      owns_gpu_data_ = true;
      owns_gpu_vals_ = true;
      return 0;  
    }
    return -1;
  }

  int matrix::Csr::copyData(memory::MemorySpace memspaceOut)
  {
    using namespace ReSolve::memory;

    index_type nnz_current = nnz_;
    if (is_expanded_) {
      nnz_current = nnz_expanded_;
    }

    switch (memspaceOut) {
      case HOST:
        //check if we need to copy or not
        if ((d_data_updated_ == true) && (h_data_updated_ == false)) {
          if (h_row_data_ == nullptr) {
            h_row_data_ = new index_type[n_ + 1];      
          }
          if (h_col_data_ == nullptr) {
            h_col_data_ = new index_type[nnz_current];      
          }
          if (h_val_data_ == nullptr) {
            h_val_data_ = new real_type[nnz_current];      
          }
          mem_.copyArrayDeviceToHost(h_row_data_, d_row_data_,      n_ + 1);
          mem_.copyArrayDeviceToHost(h_col_data_, d_col_data_, nnz_current);
          mem_.copyArrayDeviceToHost(h_val_data_, d_val_data_, nnz_current);
          h_data_updated_ = true;
          owns_cpu_data_ = true;
          owns_cpu_vals_ = true;
        }
        return 0;
      case DEVICE:
        if ((d_data_updated_ == false) && (h_data_updated_ == true)) {
          if (d_row_data_ == nullptr) {
            mem_.allocateArrayOnDevice(&d_row_data_, n_ + 1); 
          }
          if (d_col_data_ == nullptr) {
            mem_.allocateArrayOnDevice(&d_col_data_, nnz_current); 
          }
          if (d_val_data_ == nullptr) {
            mem_.allocateArrayOnDevice(&d_val_data_, nnz_current); 
          }
          mem_.copyArrayHostToDevice(d_row_data_, h_row_data_,      n_ + 1);
          mem_.copyArrayHostToDevice(d_col_data_, h_col_data_, nnz_current);
          mem_.copyArrayHostToDevice(d_val_data_, h_val_data_, nnz_current);
          d_data_updated_ = true;
          owns_gpu_data_ = true;
          owns_gpu_vals_ = true;
        }
        return 0;
      default:
        return -1;
    } // switch
  }

  int matrix::Csr::updateFromCoo(matrix::Coo* A_coo, memory::MemorySpace memspaceOut)
  {
    assert(n_            == A_coo->getNumRows());
    assert(m_            == A_coo->getNumColumns());
    assert(nnz_          == A_coo->getNnz());
    assert(is_symmetric_ == A_coo->symmetric()); // <- Do we need to check for this?

    return coo2csr(A_coo, memspaceOut);
  }


  int matrix::Csr::coo2csr(matrix::Coo* A_coo, memory::MemorySpace memspace)
  {
    //count nnzs first
    index_type nnz_unpacked = 0;
    index_type nnz = A_coo->getNnz();
    index_type n = A_coo->getNumRows();
    bool symmetric = A_coo->symmetric();
    bool expanded = A_coo->expanded();

    index_type* nnz_counts =  new index_type[n];
    std::fill_n(nnz_counts, n, 0);
    index_type* coo_rows = A_coo->getRowData(memory::HOST);
    index_type* coo_cols = A_coo->getColData(memory::HOST);
    real_type* coo_vals  = A_coo->getValues( memory::HOST);

    index_type* diag_control = new index_type[n]; //for DEDUPLICATION of the diagonal
    std::fill_n(diag_control, n, 0);
    index_type nnz_unpacked_no_duplicates = 0;
    index_type nnz_no_duplicates = nnz;


    //maybe check if they exist?
    for (index_type i = 0; i < nnz; ++i)
    {
      nnz_counts[coo_rows[i]]++;
      nnz_unpacked++;
      nnz_unpacked_no_duplicates++;
      if ((coo_rows[i] != coo_cols[i])&& (symmetric) && (!expanded))
      {
        nnz_counts[coo_cols[i]]++;
        nnz_unpacked++;
        nnz_unpacked_no_duplicates++;
      }
      if (coo_rows[i] == coo_cols[i]){
        if (diag_control[coo_rows[i]] > 0) {
         //duplicate
          nnz_unpacked_no_duplicates--;
          nnz_no_duplicates--;
        }
        diag_control[coo_rows[i]]++;
      }
    }
    this->setExpanded(true);
    this->setNnzExpanded(nnz_unpacked_no_duplicates);
    index_type* csr_ia = new index_type[n+1];
    std::fill_n(csr_ia, n + 1, 0);
    index_type* csr_ja = new index_type[nnz_unpacked];
    real_type* csr_a = new real_type[nnz_unpacked];
    index_type* nnz_shifts = new index_type[n];
    std::fill_n(nnz_shifts, n , 0);

    IndexValuePair* tmp = new IndexValuePair[nnz_unpacked]; 

    csr_ia[0] = 0;

    for (index_type i = 1; i < n + 1; ++i){
      csr_ia[i] = csr_ia[i - 1] + nnz_counts[i - 1] - (diag_control[i-1] - 1);
    }

    int r, start;


    for (index_type i = 0; i < nnz; ++i){
      //which row
      r = coo_rows[i];
      start = csr_ia[r];

      if ((start + nnz_shifts[r]) > nnz_unpacked) {
        out::warning() << "index out of bounds (case 1) start: " << start << "nnz_shifts[" << r << "] = " << nnz_shifts[r] << std::endl;
      }
      if ((r == coo_cols[i]) && (diag_control[r] > 1)) {//diagonal, and there are duplicates
        bool already_there = false;  
        for (index_type j = start; j < start + nnz_shifts[r]; ++j)
        {
          index_type c = tmp[j].getIdx();
          if (c == r) {
            real_type val = tmp[j].getValue();
            val += coo_vals[i];
            tmp[j].setValue(val);
            already_there = true;
            out::warning() << " duplicate found, row " << c << " adding in place " << j << " current value: " << val << std::endl;
          }  
        }  
        if (!already_there){ // first time this duplicates appears

          tmp[start + nnz_shifts[r]].setIdx(coo_cols[i]);
          tmp[start + nnz_shifts[r]].setValue(coo_vals[i]);

          nnz_shifts[r]++;
        }
      } else {//not diagonal
        tmp[start + nnz_shifts[r]].setIdx(coo_cols[i]);
        tmp[start + nnz_shifts[r]].setValue(coo_vals[i]);
        nnz_shifts[r]++;

        if ((coo_rows[i] != coo_cols[i]) && (symmetric == 1))
        {
          r = coo_cols[i];
          start = csr_ia[r];

          if ((start + nnz_shifts[r]) > nnz_unpacked)
            out::warning() << "index out of bounds (case 2) start: " << start << "nnz_shifts[" << r << "] = " << nnz_shifts[r] << std::endl;
          tmp[start + nnz_shifts[r]].setIdx(coo_rows[i]);
          tmp[start + nnz_shifts[r]].setValue(coo_vals[i]);
          nnz_shifts[r]++;
        }
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

    for (index_type i = 0; i < nnz_unpacked; ++i)
    {
      csr_ja[i] = tmp[i].getIdx();
      csr_a[i] = tmp[i].getValue();
    }
#if 0
    for (int i = 0; i<n; ++i){
      printf("Row: %d \n", i);
      for (int j = csr_ia[i]; j<csr_ia[i+1]; ++j){
        printf("(%d %16.16f) ", csr_ja[j], csr_a[j]);
      }
      printf("\n");
    }
#endif
    this->setNnz(nnz_no_duplicates);
    this->updateData(csr_ia, csr_ja, csr_a, memory::HOST, memspace);
    // if (memspace == "cpu"){
    //   this->updateData(csr_ia, csr_ja, csr_a, memory::HOST, memory::HOST);
    // } else {
    //   if (memspace == "cuda"){      
    //     this->updateData(csr_ia, csr_ja, csr_a, memory::HOST, memory::DEVICE);
    //   } else if (memspace == "hip"){      
    //     this->updateData(csr_ia, csr_ja, csr_a, memory::HOST, memory::DEVICE);
    //   } else {
    //     //display error
    //   }
    // }
    delete [] nnz_counts;
    delete [] tmp;
    delete [] nnz_shifts;
    delete [] csr_ia;
    delete [] csr_ja;
    delete [] csr_a;
    delete [] diag_control; 

    return 0;
  }

} // namespace ReSolve 

