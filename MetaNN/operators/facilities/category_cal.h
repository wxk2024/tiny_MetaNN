//
// Created by wxk on 2024/10/26.
//

#ifndef CATEGORY_CAL_H
#define CATEGORY_CAL_H
#include <MetaNN/data/facilities/traits.h>
#include <tuple>
namespace MetaNN{
    /* 提供运算结果类别(matrix,batch,scalar)信息 */
    template <typename TOpTag, typename THeadCate, typename...TRemainCate>
    struct OperCategory_;
    namespace NSOperCateCal {
        /// Same category check
        template <typename TCate, typename...TRemain>
        struct SameCate_
        {
            constexpr static bool value = true;
        };

        template <typename TCate, typename TCur, typename...TRemain>
        struct SameCate_<TCate, TCur, TRemain...>
        {
            constexpr static bool tmp = std::is_same<TCate, TCur>::value;
            constexpr static bool value = AndValue<tmp,
                                                   SameCate_<TCate, TRemain...>>;
        };
        template <typename TCateCont, typename...TData>
        struct Data2Cate_
        {
            using type = TCateCont;  // 包含所有类型的 tuple
        };

        template <typename...TProcessed, typename TCur, typename...TRemain>
        struct Data2Cate_<std::tuple<TProcessed...>, TCur, TRemain...>
        {
            using tmp1 = DataCategory<TCur>;
            using tmp2 = std::tuple<TProcessed..., tmp1>;
            using type = typename Data2Cate_<tmp2, TRemain...>::type;
        };
        template <typename THead, typename...TRemain>
        using Data2Cate = typename Data2Cate_<std::tuple<>, THead, TRemain...>::type;

        // 通过 操作类型和输入类型 来 得到 输出类型
        template <typename TOpTag, typename TCateContainer>
        struct CateInduce_;

        template <typename TOpTag, typename...TCates>
        struct CateInduce_<TOpTag, std::tuple<TCates...>>
        {
            using type = typename OperCategory_<TOpTag, TCates...>::type;  // 这里是一个扩展点，针对不同的 操作类型TOpTag，可以有不同的特化
        };
    }

    // 针对不同的操作判断输出类别：默认实现就是判断相等不相等
    // 当然，也存在特殊的运算，导致 OperCategory_ 的默认版本行为是错误的
    template <typename TOpTag, typename THeadCate, typename...TRemainCate>
    struct OperCategory_
    {
        static_assert(NSOperCateCal::SameCate_<THeadCate, TRemainCate...>::value,
                      "Data category mismatch.");
        using type = THeadCate;
    };
    // 推断运算结果的类别
    template <typename TOpTag, typename THead, typename...TRemain>
    using OperCateCal = typename NSOperCateCal::CateInduce_<TOpTag,
                                                            NSOperCateCal::Data2Cate<THead, TRemain...>>::type;
}
#endif //CATEGORY_CAL_H
