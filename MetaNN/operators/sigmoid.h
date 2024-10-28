//
// Created by wxk on 2024/10/28.
//

#ifndef SIGMOID_H
#define SIGMOID_H
#include <type_traits>
#include <MetaNN/operators/operators.h>
#include <MetaNN/operators/facilities/tags.h>
#include <cmath>
namespace MetaNN{

    template <typename TP>
    struct OperSigmoid_
    {
        // valid check
    private:
        using rawM = RemConstRef<TP>;

    public:
        static constexpr bool valid = IsMatrix<rawM> || IsBatchMatrix<rawM>;

    public:
        static auto Eval(TP&& p_m)
        {
            using ResType = UnaryOp<UnaryOpTags::Sigmoid, rawM>;
            return ResType(std::forward<TP>(p_m));
        }
    };
    template <typename TP,
              std::enable_if_t<OperSigmoid_<TP>::valid>* = nullptr>
    auto Sigmoid(TP&& p_m)
    {
        return OperSigmoid_<TP>::Eval(std::forward<TP>(p_m));
    }
}
#endif //SIGMOID_H
