//
// Created by wxk on 2024/11/6.
//

#ifndef ELEMENT_MUL_LAYER_H
#define ELEMENT_MUL_LAYER_H
#include <MetaNN/facilities/var_type_dict.h>
#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/policies/policy_operations.h>
#include <stdexcept>

namespace MetaNN {
    // -------------------ElementMulLayer-------------------
    // 功能：接收两个矩阵或矩阵列表，返回他们相乘的结果
    // 输入容器：ElementMulLayerInput
    // 输出容器：LayerIO
    using ElementMulLayerInput = VarTypeDict<struct ElementMulLayerIn1,
        struct ElementMulLayerIn2>;

    template<typename TPolicies>
    class ElementMulLayer {
        static_assert(IsPolicyContainer<TPolicies>, "TPolicies is not a policy container.");
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<FeedbackPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = false;
        using InputType = ElementMulLayerInput;
        using OutputType = LayerIO;

    public:
        template<typename TIn>
        auto FeedForward(const TIn &p_in) {
            const auto &val1 = p_in.template Get<ElementMulLayerIn1>();
            const auto &val2 = p_in.template Get<ElementMulLayerIn2>();

            using rawType1 = std::decay_t<decltype(val1)>;
            using rawType2 = std::decay_t<decltype(val2)>;

            static_assert(!std::is_same<rawType1, NullParameter>::value, "parameter1 is invalid");
            static_assert(!std::is_same<rawType2, NullParameter>::value, "parameter2 is invalid");

            if constexpr (IsFeedbackOutput) {
                // 如果需要反向传播时输出梯度，那么就需要保存下
                m_data1.push(MakeDynamic(val1));
                m_data2.push(MakeDynamic(val2));
            }
            return LayerIO::Create().template Set<LayerIO>(val1 * val2); // 保存相乘的结果
        }

        template<typename TGrad>
        auto FeedBackward(const TGrad &p_grad) {
            if constexpr (IsFeedbackOutput) {
                if ((m_data1.empty()) || (m_data2.empty())) {
                    throw std::runtime_error("Cannot do FeedBackward for ElementMulLayer.");
                }

                auto top1 = m_data1.top();
                auto top2 = m_data2.top();
                m_data1.pop();
                m_data2.pop();

                auto grad_eval = p_grad.template Get<LayerIO>();

                return ElementMulLayerInput::Create() //进来啥结构，出去还是啥结构
                        .template Set<ElementMulLayerIn1>(grad_eval * top2)
                        .template Set<ElementMulLayerIn2>(grad_eval * top1);
            } else {
                return ElementMulLayerInput::Create();    // 如果不需要向外传播梯度，那么直接返回 ElementMulLayerInput 空容器就可以
            }
        }

        void NeutralInvariant() {
            if constexpr (IsFeedbackOutput) {
                if ((!m_data1.empty()) || (!m_data2.empty())) {
                    // 只要还有元素那就是中性检测失败了
                    throw std::runtime_error("NeutralInvariant Fail!");
                }
            }
        }

    private:
        // 存储的中间结果类型
        // 1.如果不需要向外输出梯度信息IsFeedbackOutput=false,那么LayerInternalBuf将返回 NullParameter 类型，
        // 仅仅是占位的作用，并不会实际参与计算
        // 2.如果需要向外输出梯度，那么LayerInternalBuf会构造出 std::stack<DynamicData<...>>的数据类型
        // std::stack 中保存的是矩阵还是矩阵列表由 BatchMode 的值确定
        using DataType = LayerTraits::LayerInternalBuf<IsFeedbackOutput,
            PolicySelect<InputPolicy, CurLayerPolicy>::BatchMode,
            typename PolicySelect<OperandPolicy, CurLayerPolicy>::Element,
            typename PolicySelect<OperandPolicy, CurLayerPolicy>::Device,
            CategoryTags::Matrix, CategoryTags::BatchMatrix>;
        DataType m_data1;
        DataType m_data2;
    };
}
#endif //ELEMENT_MUL_LAYER_H
