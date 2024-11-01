//
// Created by wxk on 2024/11/1.
//

#ifndef NEGATIVE_LOG_LIKELIHOOD_H
#define NEGATIVE_LOG_LIKELIHOOD_H
#include <type_traits>
#include <vector>
#include <cmath>
#include <MetaNN/operators/operators.h>
namespace MetaNN{
    template <typename TP1, typename TP2>
    struct OperNegativeLogLikelihood_
    {
        // valid check
    private:
        using rawM1 = RemConstRef<TP1>;
        using rawM2 = RemConstRef<TP2>;

    public:
        static constexpr bool valid = (IsMatrix<rawM1> && IsMatrix<rawM2>) ||
                                      (IsBatchMatrix<rawM1> && IsBatchMatrix<rawM2>);

    public:
        static auto Eval(TP1&& p_m1, TP2&& p_m2)
        {
            static_assert(std::is_same<typename rawM1::ElementType, typename rawM2::ElementType>::value,
                          "Matrices with different element types cannot do NegativeLogLikelihood directly");
            static_assert(std::is_same<typename rawM1::DeviceType, typename rawM2::DeviceType>::value,
                          "Matrices with different device types cannot do NegativeLogLikelihood directly");

            using ResType = BinaryOp<BinaryOpTags::NegativeLogLikelihood, rawM1, rawM2>;
            return ResType(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));
        }
    };

    template <typename TP1, typename TP2,
              std::enable_if_t<OperNegativeLogLikelihood_<TP1, TP2>::valid>* = nullptr>
    auto NegativeLogLikelihood(TP1&& p_tar, TP2&& p_pre)
    {
        return OperNegativeLogLikelihood_<TP1, TP2>::Eval(std::forward<TP1>(p_tar), std::forward<TP2>(p_pre));
    }
}
#endif //NEGATIVE_LOG_LOOKLIKE_H
