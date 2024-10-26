//
// Created by wxk on 2024/10/23.
//

#ifndef BATCH_H
#define BATCH_H
#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/data/facilities/traits.h>
namespace MetaNN
{
    // 用于表示基本的 列表类型！！特殊的列表类型是 Array 与 Duplicate
    // 包括了：矩阵列表类，以及 标量列表类
    template<typename TElement, typename TDevice, typename TCategory>
    class Batch;

    template <typename TElement, typename TDevice>
    constexpr bool IsBatchMatrix<Batch<TElement, TDevice, CategoryTags::Matrix>> = true;

    template <typename TElement, typename TDevice>
    constexpr bool IsBatchScalar<Batch<TElement, TDevice, CategoryTags::Scalar>> = true;
}
#endif //BATCH_H
