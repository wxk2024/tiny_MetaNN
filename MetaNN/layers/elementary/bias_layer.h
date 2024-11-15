//
// Created by wxk on 2024/11/6.
//

#ifndef BIAS_LAYER_H
#define BIAS_LAYER_H
#include <MetaNN/data/dynamic.h>
#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/operators/collapse.h>
#include <MetaNN/model_rel/param_initializer/facilities/traits.h>
#include <string>
#include <ostream>

namespace MetaNN {
    // --------------------------BiasLayer-----------
    // 功能：接收一个矩阵或向量，计算它与层内部的矩阵或向量的和并输出。层的内部包含参数矩阵，如果需要的话，会使用 InitPolicy::BiasTypeCate所
    // 指定的初始化器来初始化该矩阵
    // 输入输出：LayerIO
    template<typename TPolicies>
    class BiasLayer {
        static_assert(IsPolicyContainer<TPolicies>, "Parameter is not policy container.");
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<FeedbackPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = PolicySelect<FeedbackPolicy, CurLayerPolicy>::IsUpdate;
        using InputType = LayerIO;
        using OutputType = LayerIO;

    private:
        using ElementType = typename PolicySelect<OperandPolicy, CurLayerPolicy>::Element;
        using DeviceType = typename PolicySelect<OperandPolicy, CurLayerPolicy>::Device;

    public:
        BiasLayer(std::string p_name, size_t p_vecLen)
            : m_name(std::move(p_name)) {
            if (p_vecLen == 0) {
                throw std::runtime_error("Invalidate input len for bias layer");
            }

            m_rowNum = 1;
            m_colNum = p_vecLen;
        }

        BiasLayer(std::string p_name, size_t p_rowNum, size_t p_colNum)
            : m_name(std::move(p_name))
              , m_rowNum(p_rowNum)
              , m_colNum(p_colNum) {
            if ((m_rowNum == 0) || (m_colNum == 0)) {
                throw std::runtime_error("Invalidate row/col num for bias layer.");
            }
        }

    public:
        template<typename TInitializer, typename TBuffer,
            typename TInitPolicies = typename TInitializer::PolicyCont>
        void Init(TInitializer &initializer, TBuffer &loadBuffer /*共享的字典*/, std::ostream *log = nullptr) {
            if (auto cit = loadBuffer.find(m_name); cit != loadBuffer.end()) {
                const Matrix<ElementType, DeviceType> &m = cit->second;
                if ((m.RowNum() != m_rowNum) || (m.ColNum() != m_colNum)) {
                    throw std::runtime_error("Load matrix error in BiasLayer");
                }
                m_bias = m;
                if (log) {
                    std::string logInfo = "Load from load buffer: " + m_name + '\n';
                    (*log) << logInfo;
                }
                return;
            } else if (initializer.IsMatrixExist(m_name)) {
                m_bias = Matrix<ElementType, DeviceType>(m_rowNum, m_colNum);
                initializer.GetMatrix(m_name, m_bias); // 用初始化器中的参数，深拷贝初始化，不共享
                loadBuffer[m_name] = m_bias;
                if (log) {
                    std::string logInfo = "Copy from initializer: " + m_name + '\n';
                    (*log) << logInfo;
                }
                return;
            } else {
                m_bias = Matrix<ElementType, DeviceType>(m_rowNum, m_colNum);
                using CurInitializer = PickInitializer<TInitPolicies, InitPolicy::BiasTypeCate>;
                if constexpr (!std::is_same<CurInitializer, void>::value) {
                    size_t fan_io = m_rowNum * m_colNum;
                    auto &cur_init = initializer.template GetFiller<CurInitializer>(); // 得到一组规则初始化器用它来初始化
                    cur_init.Fill(m_bias, fan_io, fan_io); // BiasLayer 的输入元素与输出元素个数相同
                    loadBuffer[m_name] = m_bias;
                    if (log) {
                        std::string logInfo = "Random init from initializer: " + m_name + '\n';
                        (*log) << logInfo;
                    }
                } else {
                    throw std::runtime_error("Cannot get initializer for InitPolicy::BiasTypeCate");
                }
            }
        }

        template<typename TSave>
        void SaveWeights(TSave &saver) const {
            auto cit = saver.find(m_name);
            if ((cit != saver.end()) && (cit->second != m_bias /*同名且指向不同的内存*/)) {
                throw std::runtime_error("Duplicate save for matrix: " + m_name);
            }
            saver[m_name] = m_bias;
        }

        template<typename TIn>
        auto FeedForward(const TIn &p_in) {
            const auto &val = p_in.template Get<LayerIO>();
            return LayerIO::Create().template Set<LayerIO>(val + m_bias);
        }

        template<typename TGrad>
        auto FeedBackward(const TGrad &p_grad) {
            if constexpr (IsUpdate) {
                // 如果该层需要更新
                const auto &tmp = p_grad.template Get<LayerIO>();
                assert((tmp.RowNum() == m_bias.RowNum()) && (tmp.ColNum() == m_bias.ColNum()));

                m_grad.push(MakeDynamic(tmp));
            }
            if constexpr (IsFeedbackOutput)
                return p_grad; // 继续向后传播
            else
                return LayerIO::Create();
        }

        // 用于在反向传播完成后收集参数梯度，用于后续的参数更新
        template<typename TGradCollector>
        void GradCollect(TGradCollector &col) {
            if constexpr (IsUpdate) {
                LayerTraits::MatrixGradCollect(m_bias, m_grad, col);
            }
        }

        void NeutralInvariant() const {
            if constexpr (IsUpdate) {
                if (!m_grad.empty()) {
                    throw std::runtime_error("NeutralInvariant Fail!");
                }
            }
        }

    private:
        const std::string m_name;
        size_t m_rowNum;
        size_t m_colNum;

        Matrix<ElementType, DeviceType> m_bias;

        // 与 ElementMulLayer 类型，BiasLayer 也是用了 LayerInternalBuf 来确定中间变量的数据
        // 但是 BiasLayer 使用IsUpdate 而不是 IsFeedbackOutput 来作为这个元函数的第一个参数
        // 这是因为只有在需要进行参数更新时，才需要保存计算的中间结果
        using DataType = LayerTraits::LayerInternalBuf<IsUpdate,
            PolicySelect<InputPolicy, CurLayerPolicy>::BatchMode,
            ElementType, DeviceType,
            CategoryTags::Matrix, CategoryTags::BatchMatrix>;
        DataType m_grad;
    };
}

#endif //BIAS_LAYER_H
