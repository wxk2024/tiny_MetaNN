//
// Created by wxk on 2024/10/19.
//
#include <catch2/catch_test_macros.hpp>
#include <MetaNN/policies/policy_container.h>
#include <MetaNN/facilities/traits.h>
#include <MetaNN/policies/policy_selector.h>
#include <cmath>
#include <set>
#include <type_traits>
using namespace MetaNN;
namespace {
    // 一个 Policy 组
    struct AccPolicy{
        using MajorClass = AccPolicy;

        struct AccuTypeCate
        {
            struct Add;
            struct Mul;
        };
        using Accu = AccuTypeCate::Add;

        struct IsAveValueCate;
        static constexpr bool IsAve = false;

        struct ValueTypeCate;
        using Value = float;
    };

    // Policy 模板：使用 policy 对象的
    template <typename...TPolicies>
    struct Accumulator
    {
        using TPoliCont = PolicyContainer<TPolicies...>;
        using TPolicyRes = PolicySelect<AccPolicy, TPoliCont>;

        using ValueType = typename TPolicyRes::Value;
        static constexpr bool is_ave = TPolicyRes::IsAve;
        using AccuType = typename TPolicyRes::Accu;

    public:
        template <typename TIn>
        static auto Eval(const TIn& in)
        {
            if constexpr(std::is_same<AccuType, AccPolicy::AccuTypeCate::Add>::value)
            {
                ValueType count = 0;
                ValueType res = 0;
                for (const auto& x : in)
                {
                    res += x;
                    count += 1;
                }

                if constexpr (is_ave)
                    return res / count;
                else
                    return res;
            }
            else if constexpr (std::is_same<AccuType, AccPolicy::AccuTypeCate::Mul>::value)
            {
                ValueType res = 1;
                ValueType count = 0;
                for (const auto& x : in)
                {
                    res *= x;
                    count += 1;
                }
                if constexpr (is_ave)
                    return pow(res, 1.0 / count);
                else
                    return res;
            }
            else
            {
                static_assert(DependencyFalse<AccuType>);
            }
        }
    };
}

// 从上面声明的 Policy 组中挑选生成具体的 policy 对象
#include <MetaNN/policies/policy_macro_begin.h>
TypePolicyObj (PAddAccu,     AccPolicy, Accu,  Add);
TypePolicyObj (PMulAccu,     AccPolicy, Accu,  Mul);
ValuePolicyObj(PAve,         AccPolicy, IsAve, true);
ValuePolicyObj(PNoAve,       AccPolicy, IsAve, false);
TypePolicyTemplate (PValueTypeIs,  AccPolicy, Value);
ValuePolicyTemplate(PAvePolicyIs, AccPolicy, IsAve);
#include <MetaNN/policies/policy_macro_end.h>

TEST_CASE("Policies Selector", "[policies]") {
    const int a[] ={1,2,3,4,5};
    REQUIRE(fabs(Accumulator<>::Eval(a) - 15) < 0.0001);
    REQUIRE(fabs(Accumulator<PMulAccu>::Eval(a) - 120) < 0.0001);
    REQUIRE(fabs(Accumulator<PMulAccu, PAve>::Eval(a) - pow(120.0, 0.2)) < 0.0001);
    REQUIRE(fabs(Accumulator<PAve, PMulAccu>::Eval(a) - pow(120.0, 0.2)) < 0.0001);
    REQUIRE(fabs(Accumulator<PAvePolicyIs<true>, PMulAccu>::Eval(a) - pow(120.0, 0.2)) < 0.0001);
}

unsigned int Factorial( unsigned int number ) {
    return number <= 1 ? number : Factorial(number-1)*number;
}

TEST_CASE( "Factorials are computed", "[factorial]" ) {
    REQUIRE( Factorial(1) == 1 );
    REQUIRE( Factorial(2) == 2 );
    REQUIRE( Factorial(3) == 6 );
    REQUIRE( Factorial(10) == 3628800 );
}