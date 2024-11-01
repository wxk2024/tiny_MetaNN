//
// Created by wxk on 2024/10/30.
//

#ifndef COLLAPSE_H
#define COLLAPSE_H
#include <MetaNN/data/scalar.h>
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/data/matrixs/trival_matrix.h>
#include <MetaNN/operators/facilities/tags.h>
#include <MetaNN/operators/operators.h>
#include <cassert>
#include <type_traits>
#include <utility>

namespace MetaNN {
    /*折叠 collapse 操作就是将矩阵列表求和成一个矩阵 */
    // 特化一下 运算结果类别
    template<>
    struct OperCategory_<UnaryOpTags::Collapse, CategoryTags::BatchMatrix> {
        using type = CategoryTags::Matrix;
    };

    template<typename TP>
    struct OperCollapse_ {
        // valid check
    private:
        using rawM = std::decay_t<TP>;

    public:
        static constexpr bool valid = IsBatchMatrix<rawM>;

    public:
        static auto Eval(TP &&p_m) {
            using ResType = UnaryOp<UnaryOpTags::Collapse, rawM>;
            return ResType(std::forward<TP>(p_m));
        }
    };

    template<typename TP,
        std::enable_if_t<OperCollapse_<TP>::valid>* = nullptr>
    auto Collapse(TP &&p_m) {
        return OperCollapse_<TP>::Eval(std::forward<TP>(p_m));
    }
}
#endif //COLLAPSE_H
