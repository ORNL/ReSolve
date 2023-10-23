#pragma once


namespace ReSolve {

  /// @brief Helper class for COO matrix sorting
  class IndexValuePair
  {
    public:
      IndexValuePair() : idx_(0), value_(0.0)
      {}
      ~IndexValuePair()
      {}
      void setIdx (index_type new_idx)
      {
        idx_ = new_idx;
      }
      void setValue (real_type new_value)
      {
        value_ = new_value;
      }

      index_type getIdx()
      {
        return idx_;
      }
      real_type getValue()
      {
        return value_;
      }

      bool operator < (const IndexValuePair& str) const
      {
        return (idx_ < str.idx_);
      }  

    private:
      index_type idx_;
      real_type value_;
  };

} // namespace ReSolve

