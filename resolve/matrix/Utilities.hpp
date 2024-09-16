#pragma once

#include <list>
#include <resolve/MemoryUtils.hpp>
#include <resolve/utilities/logger/Logger.hpp>

namespace ReSolve
{
  /// @brief Helper class for COO matrix sorting
  class CooTriplet
  {
    public:
      CooTriplet() : rowidx_(0), colidx_(0), value_(0.0)
      {}
      CooTriplet(index_type i, index_type j, real_type v) : rowidx_(i), colidx_(j), value_(v)
      {}
      ~CooTriplet()
      {}
      void setColIdx (index_type new_idx)
      {
        colidx_ = new_idx;
      }
      void setValue (real_type new_value)
      {
        value_ = new_value;
      }
      void set(index_type rowidx, index_type colidx, real_type value)
      {
        rowidx_ = rowidx;
        colidx_ = colidx;
        value_  = value;
      }

      index_type getRowIdx()
      {
        return rowidx_;
      }
      index_type getColIdx()
      {
        return colidx_;
      }
      real_type getValue()
      {
        return value_;
      }

      bool operator < (const CooTriplet& str) const
      {
        if (rowidx_ < str.rowidx_)
          return true;

        if ((rowidx_ == str.rowidx_) && (colidx_ < str.colidx_))
          return true;

        return false;
      }

      bool operator == (const CooTriplet& str) const
      {
        return (rowidx_ == str.rowidx_) && (colidx_ == str.colidx_);
      }

      CooTriplet& operator += (const CooTriplet t)
      {
        if ((rowidx_ != t.rowidx_) || (colidx_ != t.colidx_)) {
          io::Logger::error() << "Adding values into non-matching triplet.\n";
        }
        value_ += t.value_;
        return *this;
      }

      void print() const
      {
        // Add 1 to indices to restore indexing from MM format
        std::cout << rowidx_ << " " << colidx_ << " " << value_ << "\n";
      }

    private:
      index_type rowidx_{0};
      index_type colidx_{0};
      real_type value_{0.0};
  };

  inline void print_list(std::list<CooTriplet>& l)
  {
    // Print out the list
    std::cout << "tmp list:\n";
    for (CooTriplet& n : l)
      n.print();
    std::cout << "\n";
  }

  namespace matrix
  {
    // Forward declarations
    class Coo;
    class Csr;

    /// @brief Converts symmetric or general COO to general CSR matrix
    int coo2csr(matrix::Coo* A_coo, matrix::Csr* A_csr, memory::MemorySpace memspace);

    /// @brief Converts symmetric or general COO to general CSR matrix
    int coo2csr_new(matrix::Coo* A_coo, matrix::Csr* A_csr, memory::MemorySpace memspace);
  }
}
