//
// Created by wxk on 2024/10/22.
//
#include <catch2/catch_test_macros.hpp>
#include <MetaNN/data/scalar.h>
#include <MetaNN/data/matrixs/cpu_matrix.h>
#include <MetaNN/data/matrixs/trival_matrix.h>
#include <MetaNN/data/matrixs/zero_matrix.h>
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
