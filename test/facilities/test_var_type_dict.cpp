//
// Created by wxk on 2024/10/20.
//
#include <iostream>
#include <cassert>
#include <cmath>
#include <set>
#include <MetaNN/facilities/var_type_dict.h>

using namespace std;
using namespace MetaNN;
namespace{
    struct A; struct B; struct Weight; struct XX;

    struct FParams : public VarTypeDict<struct A,
                                        struct B,
                                        struct Weight> {};
    template <typename TIn>
    float fun(const TIn& in) {
        auto a = in.template Get<A>();
        auto b = in.template Get<B>();
        auto weight = in.template Get<Weight>();

        return a * weight + b * (1 - weight);
    }
}

#include <catch2/catch_test_macros.hpp>
TEST_CASE("test type dict","[type dict]") {
    float a = 1.3f;

    auto res = fun(FParams::Create()
                             .Set<A>(a)
                             .Set<B>(2.4f)
                             .Set<Weight>(0.1f));

    REQUIRE(fabs(res-0.1*1.3-0.9*2.4)<0.0001);
    using pp = int;
}
///


