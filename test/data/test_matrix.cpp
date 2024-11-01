//
// Created by wxk on 2024/10/22.
//
#include <catch2/catch_test_macros.hpp>
#include <MetaNN/data/scalar.h>
#include <MetaNN/data/matrixs/cpu_matrix.h>
#include <MetaNN/data/matrixs/trival_matrix.h>
#include <MetaNN/data/matrixs/zero_matrix.h>
#include <MetaNN/data/matrixs/one_hot_vector.h>
#include <MetaNN/data/batch/matrix.h>
#include <MetaNN/data/batch/duplicate.h>
#include <MetaNN/data/batch/scalar.h>
#include <MetaNN/operators/facilities/category_cal.h>
#include <MetaNN/operators/facilities/organizer.h>
#include <MetaNN/operators/facilities/oper_seq.h>
#include <MetaNN/operators/operators.h>
#include <MetaNN/operators/sigmoid.h>
#include <MetaNN/operators/add.h>
#include <MetaNN/operators/transpose.h>
#include <MetaNN/operators/collapse.h>
#include <MetaNN/operators/divide.h>
#include <MetaNN/operators/abs.h>
#include <MetaNN/operators/dot.h>
#include <MetaNN/operators/element_mul.h>
#include <MetaNN/operators/interpolate.h>
#include <MetaNN/operators/negative_log_likelihood.h>
#include <MetaNN/data/batch/array.h>
#include "../facilities/calculate_tags.h"
using namespace MetaNN;
TEST_CASE("general matrix", "[matrix]") {
    REQUIRE(IsMatrix<Matrix<CheckElement, CheckDevice>> == true);
    REQUIRE(IsMatrix<Matrix<CheckElement, CheckDevice>&> == true);
    REQUIRE(IsMatrix<Matrix<CheckElement, CheckDevice>&&> == true);
    REQUIRE(IsMatrix<const Matrix<CheckElement, CheckDevice>&> == true);
    REQUIRE(IsMatrix<const Matrix<CheckElement, CheckDevice>&&> == true);

    Matrix<CheckElement, CheckDevice> rm;
    REQUIRE(rm.RowNum() == 0);
    REQUIRE(rm.ColNum() == 0);

    rm = Matrix<CheckElement, CheckDevice>(10, 20);
    assert(rm.RowNum() == 10);
    assert(rm.ColNum() == 20);

}

TEST_CASE("trival matrix", "[matrix]") {
REQUIRE(IsMatrix<TrivalMatrix<int,CheckElement, CheckDevice>> == true);
}

TEST_CASE("zero matrix","[matrix]") {
    REQUIRE(IsMatrix<ZeroMatrix<int, CheckDevice>> == true);
    REQUIRE(IsMatrix<ZeroMatrix<int, CheckDevice>&> == true);
    REQUIRE(IsMatrix<ZeroMatrix<int, CheckDevice>&&> == true);
    REQUIRE(IsMatrix<const ZeroMatrix<int, CheckDevice>&> == true);
    REQUIRE(IsMatrix<const ZeroMatrix<int, CheckDevice>&&> == true);
}
