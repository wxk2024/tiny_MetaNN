//
// Created by wxk on 2024/11/1.
//

#ifndef TANH_H
#define TANH_H
#include <type_traits>
#include <MetaNN/operators/operators.h>
#include <cmath>
namespace MetaNN{
    template <typename TP>
    struct OperTanh_
    {
        // valid check
    private:
        using rawM = RemConstRef<TP>;

    public:
        static constexpr bool valid = IsMatrix<rawM> || IsBatchMatrix<rawM>;

    public:
        static auto Eval(TP&& p_m)
        {
            using ResType = UnaryOp<UnaryOpTags::Tanh, rawM>;
            return ResType(std::forward<TP>(p_m));
        }
    };

    template <typename TP,
              std::enable_if_t<OperTanh_<TP>::valid>* = nullptr>
    auto Tanh(TP&& p_m)
    {
        return OperTanh_<TP>::Eval(std::forward<TP>(p_m));
    }
}
#endif //TANH_H
