//
// Created by wxk on 2024/11/1.
//

#ifndef TANH_DERIVATIVE_H
#define TANH_DERIVATIVE_H
#include <type_traits>
#include <MetaNN/operators/operators.h>
#include <cmath>

namespace MetaNN {
    template<typename TGrad, typename TOut>
    struct OperTanhDerivative_ {
    private:
        using rawM1 = RemConstRef<TGrad>;
        using rawM2 = RemConstRef<TOut>;

    public:
        static constexpr bool valid = (IsMatrix<rawM1> && IsMatrix<rawM2>) ||
                                      (IsBatchMatrix<rawM1> && IsBatchMatrix<rawM2>);

    public:
        static auto Eval(TGrad &&p_grad, TOut &&p_out) {
            using ResType = BinaryOp<BinaryOpTags::TanhDerivative, rawM1, rawM2>;
            return ResType(std::forward<TGrad>(p_grad), std::forward<TOut>(p_out));
        }
    };

    template<typename TGrad, typename TOut,
        std::enable_if_t<OperTanhDerivative_<TGrad, TOut>::valid>* = nullptr>
    auto TanhDerivative(TGrad &&p_grad, TOut &&p_out) {
        return OperTanhDerivative_<TGrad, TOut>::Eval(std::forward<TGrad>(p_grad),
                                                      std::forward<TOut>(p_out));
    }
}
#endif //TANH_DERIVATIVE_H
