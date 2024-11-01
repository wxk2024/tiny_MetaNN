//
// Created by wxk on 2024/10/19.
//
#include <catch2/catch_test_macros.hpp>
#include <MetaNN/policies/policy_container.h>
#include <MetaNN/facilities/traits.h>
#include <MetaNN/policies/policy_selector.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/policies/change_policy.h>
#include <MetaNN/layers/facilities/policies.h>
#include <cmath>
#include <set>
#include <type_traits>
using namespace MetaNN;
namespace {
    // 一个 Policy 组；需要被继承，改写其中的 minor class
    struct AccPolicy{
        using MajorClass = AccPolicy;

        struct AccuTypeCate   // minor class
        {
            struct Add;
            struct Mul;
        };
        using Accu = AccuTypeCate::Add;

        struct IsAveValueCate; // minor class
        static constexpr bool IsAve = false;

        struct ValueTypeCate; // minor class
        using Value = float;
    };

    // 使用 policy 对象的提供的信息来决定它的行为
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

TEST_CASE("Policies change", "[policies]") {
    struct Tag1;
    struct Tag2;
    struct Tag3;
    using input = PolicyContainer<PBatchMode,
                                  SubPolicyContainer<Tag1, PNoBatchMode>>;

    using check1 = ChangePolicy<PEnableBptt, input>;
    REQUIRE(std::is_same<check1, PolicyContainer<PBatchMode, SubPolicyContainer<Tag1, PNoBatchMode>, PEnableBptt>>::value);

    using check2 = ChangePolicy<PNoBatchMode, input>;
    REQUIRE(std::is_same<check2, PolicyContainer<SubPolicyContainer<Tag1, PNoBatchMode>, PNoBatchMode>>::value);
}

TEST_CASE("Policies operate","[policies]") {
    struct Tag1;
    struct Tag2;
    struct Tag3;
    using input = PolicyContainer<PBatchMode,
                                      SubPolicyContainer<Tag1, PNoBatchMode,
                                                         SubPolicyContainer<Tag2>>>;
    using check1 = SubPolicyPicker<input, Tag3>;
    // 没有符合的 subpolicycontainer 所以就只剩下最外层的 PBatchMode了
    REQUIRE(std::is_same<check1, PolicyContainer<PBatchMode>>::value);

    // 找到了符合的 subpolicycontainer , 再加上最外层的 PBatchMode
    using check2 = SubPolicyPicker<input, Tag1>;
    REQUIRE(std::is_same<check2, PolicyContainer<PNoBatchMode, SubPolicyContainer<Tag2>>>::value);

    using check3 = SubPolicyPicker<check2, Tag3>;
    REQUIRE(std::is_same<check3, PolicyContainer<PNoBatchMode>>::value);

    using check4 = SubPolicyPicker<check2, Tag2>;
    REQUIRE(std::is_same<check4, PolicyContainer<PNoBatchMode>>::value);
}
