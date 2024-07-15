#pragma once

#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Utilities.hpp>
#include <tests/unit/TestBase.hpp>

using index_type = ReSolve::index_type;
using real_type = ReSolve::real_type;

namespace ReSolve
{
  namespace tests
  {
    /**
     * @class Unit tests for matrix conversions
     */
    class MatrixConversionTests : TestBase
    {
      public:
        MatrixConversionTests()
        {
        }

        virtual ~MatrixConversionTests()
        {
        }

        TestOutcome simpleUpperUnexpandedSymmetricMatrix()
        {
          TestStatus status;

          matrix::Coo A(simple_upper_unexpanded_symmetric_n_,
                        simple_upper_unexpanded_symmetric_m_,
                        simple_upper_unexpanded_symmetric_nnz_,
                        true,
                        false);
          A.updateData(simple_upper_unexpanded_symmetric_i_,
                       simple_upper_unexpanded_symmetric_j_,
                       simple_upper_unexpanded_symmetric_a_,
                       memory::HOST,
                       memory::HOST);
          ReSolve::matrix::Csr B(A.getNumRows(), A.getNumColumns(), 0);

          status *= ReSolve::matrix::coo2csr(&A, &B, memory::HOST) == 0;
          status *= this->verifyAnswer(&B,
                                       simple_symmetric_expected_n_,
                                       simple_symmetric_expected_m_,
                                       simple_symmetric_expected_nnz_,
                                       simple_symmetric_expected_i_,
                                       simple_symmetric_expected_j_,
                                       simple_symmetric_expected_a_);

          return status.report(__func__);
        }

        TestOutcome simpleLowerUnexpandedSymmetricMatrix()
        {
          TestStatus status;

          matrix::Coo A(simple_lower_unexpanded_symmetric_n_,
                        simple_lower_unexpanded_symmetric_m_,
                        simple_lower_unexpanded_symmetric_nnz_,
                        true,
                        false);
          A.updateData(simple_lower_unexpanded_symmetric_i_,
                       simple_lower_unexpanded_symmetric_j_,
                       simple_lower_unexpanded_symmetric_a_,
                       memory::HOST,
                       memory::HOST);
          ReSolve::matrix::Csr B(A.getNumRows(), A.getNumColumns(), 0);

          status *= ReSolve::matrix::coo2csr(&A, &B, memory::HOST) == 0;
          status *= this->verifyAnswer(&B,
                                       simple_symmetric_expected_n_,
                                       simple_symmetric_expected_m_,
                                       simple_symmetric_expected_nnz_,
                                       simple_symmetric_expected_i_,
                                       simple_symmetric_expected_j_,
                                       simple_symmetric_expected_a_);

          return status.report(__func__);
        }

        TestOutcome simpleMainDiagonalOnlyMatrix()
        {
          TestStatus status;

          matrix::Coo A(simple_main_diagonal_only_n_,
                        simple_main_diagonal_only_m_,
                        simple_main_diagonal_only_nnz_,
                        true,
                        false);
          A.updateData(simple_main_diagonal_only_i_j_,
                       simple_main_diagonal_only_i_j_,
                       simple_main_diagonal_only_a_,
                       memory::HOST,
                       memory::HOST);
          ReSolve::matrix::Csr B(A.getNumRows(), A.getNumColumns(), 0);

          status *= ReSolve::matrix::coo2csr(&A, &B, memory::HOST) == 0;
          status *= this->verifyAnswer(&B,
                                       simple_main_diagonal_only_n_,
                                       simple_main_diagonal_only_m_,
                                       simple_main_diagonal_only_nnz_,
                                       simple_main_diagonal_only_i_j_,
                                       simple_main_diagonal_only_i_j_,
                                       simple_main_diagonal_only_a_);

          return status.report(__func__);
        }

        TestOutcome simpleFullAsymmetricMatrix()
        {
          TestStatus status;

          matrix::Coo A(simple_asymmetric_n_,
                        simple_asymmetric_m_,
                        simple_asymmetric_nnz_);
          A.updateData(simple_asymmetric_i_,
                       simple_asymmetric_j_,
                       simple_asymmetric_a_,
                       memory::HOST,
                       memory::HOST);
          ReSolve::matrix::Csr B(A.getNumRows(), A.getNumColumns(), 0);

          status *= ReSolve::matrix::coo2csr(&A, &B, memory::HOST) == 0;
          status *= this->verifyAnswer(&B,
                                       simple_asymmetric_expected_n_,
                                       simple_asymmetric_expected_m_,
                                       simple_asymmetric_expected_nnz_,
                                       simple_asymmetric_expected_i_,
                                       simple_asymmetric_expected_j_,
                                       simple_asymmetric_expected_a_);

          return status.report(__func__);
        }

      private:
        const index_type simple_symmetric_expected_n_ = 5;
        const index_type simple_symmetric_expected_m_ = 5;
        const index_type simple_symmetric_expected_nnz_ = 8;
        index_type simple_symmetric_expected_i_[8] = {0, 1, 1, 1, 2, 3, 3, 4};
        index_type simple_symmetric_expected_j_[8] = {0, 1, 2, 3, 1, 1, 4, 3};
        real_type simple_symmetric_expected_a_[8] = {3.0, 7.0, 11.0, 7.0, 11.0, 7.0, 8.0, 8.0};

        const index_type simple_upper_unexpanded_symmetric_n_ = 5;
        const index_type simple_upper_unexpanded_symmetric_m_ = 5;
        const index_type simple_upper_unexpanded_symmetric_nnz_ = 8;
        index_type simple_upper_unexpanded_symmetric_i_[8] = {0, 0, 1, 1, 1, 1, 1, 3};
        index_type simple_upper_unexpanded_symmetric_j_[8] = {0, 0, 1, 1, 2, 2, 3, 4};
        real_type simple_upper_unexpanded_symmetric_a_[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

        const index_type simple_main_diagonal_only_n_ = 5;
        const index_type simple_main_diagonal_only_m_ = 5;
        const index_type simple_main_diagonal_only_nnz_ = 3;
        index_type simple_main_diagonal_only_i_j_[3] = {1, 3, 4};
        real_type simple_main_diagonal_only_a_[3] = {1.0, 2.0, 3.0};

        const index_type simple_lower_unexpanded_symmetric_n_ = 5;
        const index_type simple_lower_unexpanded_symmetric_m_ = 5;
        const index_type simple_lower_unexpanded_symmetric_nnz_ = 8;
        index_type simple_lower_unexpanded_symmetric_i_[8] = {0, 0, 1, 1, 2, 2, 3, 4};
        index_type simple_lower_unexpanded_symmetric_j_[8] = {0, 0, 1, 1, 1, 1, 1, 3};
        real_type simple_lower_unexpanded_symmetric_a_[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

        const index_type simple_asymmetric_n_ = 5;
        const index_type simple_asymmetric_m_ = 5;
        const index_type simple_asymmetric_nnz_ = 10;
        index_type simple_asymmetric_i_[10] = {0, 1, 1, 2, 3, 3, 3, 4, 1, 1};
        index_type simple_asymmetric_j_[10] = {0, 1, 3, 1, 1, 4, 4, 3, 2, 2};
        real_type simple_asymmetric_a_[10] = {2.0, 4.0, 7.0, 9.0, 6.0, 7.0, 8.0, 8.0, 5.0, 6.0};

        const index_type simple_asymmetric_expected_n_ = 5;
        const index_type simple_asymmetric_expected_m_ = 5;
        const index_type simple_asymmetric_expected_nnz_ = 8;
        index_type simple_asymmetric_expected_i_[8] = {0, 1, 1, 1, 2, 3, 3, 4};
        index_type simple_asymmetric_expected_j_[8] = {0, 1, 2, 3, 1, 1, 4, 3};
        real_type simple_asymmetric_expected_a_[8] = {2.0, 4.0, 11.0, 7.0, 9.0, 6.0, 15.0, 8.0};

        bool verifyAnswer(matrix::Csr* A,
                          const index_type& n,
                          const index_type& m,
                          const index_type& nnz,
                          index_type* is,
                          index_type* js,
                          real_type* as)
        {
          if (n != A->getNumRows() || m != A->getNumColumns() || nnz != A->getNnz()) {
            return false;
          }

          index_type* rows = A->getRowData(memory::HOST);
          index_type* columns = A->getColData(memory::HOST);
          real_type* values = A->getValues(memory::HOST);

          index_type answer_offset = 0;
          for (index_type i = 0; i < A->getNumRows(); i++) {
            for (index_type offset = rows[i]; offset < rows[i + 1]; offset++) {
              if (i != is[answer_offset] || columns[offset] != js[answer_offset] || !isEqual(values[offset], as[answer_offset])) {
                return false;
              }
              answer_offset++;
            }
          }

          return true;
        }
    };
  } // namespace tests
} // namespace ReSolve
