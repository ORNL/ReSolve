#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <tuple>

#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Utilities.hpp>
#include <tests/unit/TestBase.hpp>

namespace ReSolve
{
  namespace tests
  {
    class MatrixExpansionTests : TestBase
    {
      public:
        MatrixExpansionTests()
        {
        }

        virtual ~MatrixExpansionTests()
        {
        }

        TestOutcome cooMatrix5x5()
        {
          TestStatus status;

          std::unique_ptr<matrix::Coo> A = buildCooMatrix5x5();
          matrix::expand(*A);

          A->print();

          status *= validateAnswer(*A, target_triples_5x5_);

          return status.report(__func__);
        }

        TestOutcome csrMatrix5x5()
        {
          TestStatus status;

          std::unique_ptr<matrix::Csr> A = buildCsrMatrix5x5();
          matrix::expand(*A);

          A->print();

          status *= validateAnswer(*A, target_triples_5x5_);

          return status.report(__func__);
        }

      private:
        std::vector<std::tuple<index_type, index_type, real_type>>
            target_triples_5x5_ = {{1, 1, 1.0},
                                   {3, 4, 2.0},
                                   {4, 3, 2.0},
                                   {0, 4, 3.0},
                                   {4, 0, 3.0}};

        bool validateAnswer(matrix::Coo& A,
                            std::vector<std::tuple<index_type, index_type, real_type>> target)
        {
          std::shared_ptr<index_type> i(new index_type(0));
          std::shared_ptr<index_type> nnz(new index_type(A.getNnz()));
          index_type* rows = A.getRowData(memory::HOST);
          index_type* columns = A.getColData(memory::HOST);
          real_type* values = A.getValues(memory::HOST);

          if (rows == nullptr || columns == nullptr || values == nullptr) {
            return false;
          }

          return validateAnswer(
              [=]() -> std::tuple<std::tuple<index_type, index_type, index_type>, bool> {
                if (*i == *nnz) {
                  return {{0, 0, 0}, false};
                }

                (*i)++;
                return {{rows[*i - 1], columns[*i - 1], values[*i - 1]}, true};
              },
              target);
        }

        bool validateAnswer(matrix::Csr& A,
                            std::vector<std::tuple<index_type, index_type, real_type>> target)
        {
          std::shared_ptr<index_type> i(new index_type(0)), j(new index_type(0));
          std::shared_ptr<index_type> nnz(new index_type(A.getNnz()));
          index_type* rows = A.getRowData(memory::HOST);
          index_type* columns = A.getColData(memory::HOST);
          real_type* values = A.getValues(memory::HOST);

          if (rows == nullptr || columns == nullptr || values == nullptr) {
            return false;
          }

          return validateAnswer(
              [=]() -> std::tuple<std::tuple<index_type, index_type, real_type>, bool> {
                if (*j == *nnz) {
                  return {{0, 0, 0}, false};
                }

                while (rows[*i + 1] == *j) {
                  (*i)++;
                }

                (*j)++;
                return {{*i, columns[*j - 1], values[*j - 1]}, true};
              },
              target);
        }

        bool validateAnswer(std::function<std::tuple<std::tuple<index_type, index_type, real_type>, bool>()> f,
                            std::vector<std::tuple<index_type, index_type, real_type>> target)
        {
          bool ok;
          std::tuple<index_type, index_type, real_type> t;

          for (std::tie(t, ok) = f(); ok; std::tie(t, ok) = f()) {
            if (std::find(target.begin(), target.end(), t) == target.end()) {
              return false;
            }
          }

          return true;
        }

        std::vector<index_type> rows_coo_5x5_ = {1, 3, 0};
        std::vector<index_type> columns_coo_5x5_ = {1, 4, 4};
        std::vector<real_type> values_coo_5x5_ = {1.0, 2.0, 3.0};

        std::unique_ptr<matrix::Coo> buildCooMatrix5x5()
        {
          std::unique_ptr<matrix::Coo> A(new matrix::Coo(5, 5, 3, true, false));
          A->updateData(rows_coo_5x5_.data(),
                        columns_coo_5x5_.data(),
                        values_coo_5x5_.data(),
                        memory::HOST,
                        memory::HOST);

          return A;
        }

        std::vector<index_type> rows_csr_5x5_ = {0, 1, 2, 2, 3, 3};
        std::vector<index_type> columns_csr_5x5_ = {4, 1, 4};
        std::vector<real_type> values_csr_5x5_ = {3.0, 1.0, 2.0};

        std::unique_ptr<matrix::Csr> buildCsrMatrix5x5()
        {
          std::unique_ptr<matrix::Csr> A(new matrix::Csr(5, 5, 3, true, false));
          A->updateData(rows_csr_5x5_.data(),
                        columns_csr_5x5_.data(),
                        values_csr_5x5_.data(),
                        memory::HOST,
                        memory::HOST);

          return A;
        }
    };
  } // namespace tests
} // namespace ReSolve
