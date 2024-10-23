//
// Created by wxk on 2024/10/22.
//
#include <catch2/catch_test_macros.hpp>
#include <MetaNN/data/scalar.h>

using namespace std;
using namespace MetaNN;
TEST_CASE("scalar", "[scalar]") {
    REQUIRE(IsScalar<Scalar<int, DeviceTags::CPU>> == true);
    REQUIRE(IsScalar<Scalar<int, DeviceTags::CPU>&> == true);
    REQUIRE(IsScalar<Scalar<int, DeviceTags::CPU>&&> == true);
    REQUIRE(IsScalar<const Scalar<int, DeviceTags::CPU>&> == true);
    REQUIRE(IsScalar<const Scalar<int, DeviceTags::CPU>&&> == true);
    Scalar<float, DeviceTags::CPU> pi(3.1415926f);
    REQUIRE(pi == pi);
    auto x =pi.EvalRegister();
    REQUIRE(x.Data() == pi);
}