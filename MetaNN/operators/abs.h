//
// Created by wxk on 2024/11/1.
//

#ifndef ABS_H
#define ABS_H
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/data/matrices/trival_matrix.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/operators/facilities/tags.h>
#include <MetaNN/operators/operators.h>
#include <cassert>
#include <type_traits>
#include <utility>
namespace MetaNN{
    template <typename TP>
    struct OperAbs_
    {
        // valid check
    private:
        using rawM = RemConstRef<TP>;

    public:
        static constexpr bool valid = IsMatrix<rawM> || IsBatchMatrix<rawM>;

    public:
        static auto Eval(TP&& p_m)
        {
            using ResType = UnaryOp<UnaryOpTags::Abs, rawM>; // 运算模板
            return ResType(std::forward<TP>(p_m));
        }
    };

    template <typename TP,
              std::enable_if_t<OperAbs_<TP>::valid>* = nullptr>
    auto Abs(TP&& p_m)
    {
        return OperAbs_<TP>::Eval(std::forward<TP>(p_m));
    }
}
#endif //ABS_H
