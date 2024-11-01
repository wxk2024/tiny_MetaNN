//
// Created by wxk on 2024/11/1.
//

#ifndef SIGN_H
#define SIGN_H
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/data/matrixs/trival_matrix.h>
// #include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/operators/facilities/tags.h>
#include <MetaNN/operators/operators.h>
#include <cassert>
#include <type_traits>
#include <utility>
namespace MetaNN{
    template <typename TP>
    struct OperSign_
    {
        // valid check
    private:
        using rawM = RemConstRef<TP>;

    public:
        static constexpr bool valid = IsMatrix<rawM> || IsBatchMatrix<rawM>;

    public:
        static auto Eval(TP&& p_m)
        {
            using ResType = UnaryOp<UnaryOpTags::Sign, rawM>;
            return ResType(std::forward<TP>(p_m));
        }
    };

    template <typename TP,
              std::enable_if_t<OperSign_<TP>::valid>* = nullptr>
    auto Sign(TP&& p_m)
    {
        return OperSign_<TP>::Eval(std::forward<TP>(p_m));
    }
}
#endif //SIGN_H
