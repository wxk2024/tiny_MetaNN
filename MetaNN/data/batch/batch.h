//
// Created by wxk on 2024/10/23.
//

#ifndef BATCH_H
#define BATCH_H
#include <MetaNN/data/facilities/tags.h>
namespace MetaNN
{
    template<typename TElement, typename TDevice, typename TCategory>
    class Batch;

    template <typename TElement, typename TDevice>
    constexpr bool IsBatchMatrix<Batch<TElement, TDevice, CategoryTags::Matrix>> = true;

    template <typename TElement, typename TDevice>
    constexpr bool IsBatchScalar<Batch<TElement, TDevice, CategoryTags::Scalar>> = true;
}
#endif //BATCH_H
