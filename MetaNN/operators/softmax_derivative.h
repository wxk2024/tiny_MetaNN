//
// Created by wxk on 2024/11/1.
//

#ifndef SOFTMAX_DERIVATIVE_H
#define SOFTMAX_DERIVATIVE_H
#include <MetaNN/operators/operators.h>
#include <stdexcept>
namespace MetaNN {

    template <typename TGrad, typename TSOut>
    struct OperVecSoftmaxDerivative_
    {
        // valid check
    private:
        using rawGrad = RemConstRef<TGrad>;
        using rawSOut = RemConstRef<TSOut>;

    public:
        static constexpr bool valid = (IsMatrix<rawGrad> && IsMatrix<rawSOut>) ||
                                      (IsBatchMatrix<rawGrad> && IsBatchMatrix<rawSOut>);

    public:
        static auto Eval(TGrad&& p_grad, TSOut&& p_sout)
        {
            static_assert(std::is_same<typename rawGrad::ElementType, typename rawSOut::ElementType>::value,
                          "Element type mismatch.");
            static_assert(std::is_same<typename rawGrad::DeviceType, typename rawSOut::DeviceType>::value,
                          "Device type mismatch.");

            using ResType = BinaryOp<BinaryOpTags::VecSoftmaxDerivative, rawGrad, rawSOut>;
            return ResType(std::forward<TGrad>(p_grad), std::forward<TSOut>(p_sout));
        }
    };

    template <typename TGrad, typename TSOut,
              std::enable_if_t<OperVecSoftmaxDerivative_<TGrad, TSOut>::valid>* = nullptr>
    auto VecSoftmaxDerivative(TGrad&& p_grad, TSOut&& p_sout/*雅可比矩阵*/)
    {
        return OperVecSoftmaxDerivative_<TGrad, TSOut>::Eval(std::forward<TGrad>(p_grad),
                                                             std::forward<TSOut>(p_sout));
    }
}
#endif //SOFTMAX_DERIVATIVE_H
