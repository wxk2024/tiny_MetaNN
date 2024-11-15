//
// Created by wxk on 2024/11/6.
//

#ifndef ADD_LAYER_H
#define ADD_LAYER_H
#include <MetaNN/facilities/var_type_dict.h>
#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/interface_fun.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/policies/policy_operations.h>

namespace MetaNN
{
    //-------------------AddLayer--------------
    // 功能：接收两个矩阵或矩阵列表，返回他们相加的结果
    // 输入：AddLayerInput
    // 输出：LayerIO
    // 层的输入容器
    using AddLayerInput = VarTypeDict<struct AddLayerIn1, struct AddLayerIn2>;

    template <typename TPolicies>
    class AddLayer
    {
        static_assert(IsPolicyContainer<TPolicies>, "TPolicies is not a policy container.");
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<FeedbackPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = false; // 它本身没有参数，不需要更新
        using InputType = AddLayerInput;
        using OutputType = LayerIO;

    public:
        template <typename TIn>
        auto FeedForward(const TIn& p_in)
        {
            const auto& val1 = p_in.template Get<AddLayerIn1>(); // 异步词典获取方式
            const auto& val2 = p_in.template Get<AddLayerIn2>();

            using rawType1 = std::decay_t<decltype(val1)>;
            using rawType2 = std::decay_t<decltype(val2)>;

            static_assert(!std::is_same<rawType1, NullParameter>::value, "parameter1 is invalid");
            static_assert(!std::is_same<rawType2, NullParameter>::value, "parameter2 is invalid");

            return OutputType::Create().template Set<LayerIO>(val1 + val2); // 异步词典设置方式
        }

        template <typename TGrad>
        auto FeedBackward(TGrad&& p_grad)
        {
            if constexpr (IsFeedbackOutput)
            {
                auto res = p_grad.template Get<LayerIO>();
                return AddLayerInput::Create().template Set<AddLayerIn1>(res)
                                              .template Set<AddLayerIn2>(res);
            }
            else
            {
                return AddLayerInput::Create();
            }
        }
    };
}
#endif //ADD_LAYER_H
