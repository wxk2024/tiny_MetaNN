//
// Created by wxk on 2024/11/1.
//

#ifndef SOFTMAX_H
#define SOFTMAX_H
#include <type_traits>
#include <MetaNN/operators/operators.h>
#include <cmath>
#include <algorithm>
namespace MetaNN{
    template <typename TP>
    struct OperVecSoftmax_
    {
        // valid check
    private:
        using rawM = RemConstRef<TP>;

    public:
        static constexpr bool valid = IsMatrix<rawM> || IsBatchMatrix<rawM>;

    public:
        template <typename T>
        static auto Eval(TP&& p_m)
        {
            using ResType = UnaryOp<UnaryOpTags::VecSoftmax, rawM>;
            return ResType(std::forward<TP>(p_m));
        }
    };

    template <typename TP,
              std::enable_if_t<OperVecSoftmax_<TP>::valid>* = nullptr>
    auto VecSoftmax(TP&& p_m)
    {
        return OperVecSoftmax_<TP>::
                template Eval<DataCategory<TP>>(std::forward<TP>(p_m));
    }
}
#endif //SOFTMAX_H
