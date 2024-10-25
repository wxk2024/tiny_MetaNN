//
// Created by wxk on 2024/10/20.
//

#ifndef TRAITS_H
#define TRAITS_H
namespace MetaNN {
    // 元函数 : 扩展点，拿到一个类型我要知道他是什么标签
    /// is scalar
    /// is scalar
    template <typename T>
    constexpr bool IsScalar = false;

    template <typename T>
    constexpr bool IsScalar<const T> = IsScalar<T>;

    template <typename T>
    constexpr bool IsScalar<T&> = IsScalar<T>;

    template <typename T>
    constexpr bool IsScalar<T&&> = IsScalar<T>;
    /// is matrix
    template <typename T>
    constexpr bool IsMatrix = false;

    template <typename T>
    constexpr bool IsMatrix<const T> = IsMatrix<T>;

    template <typename T>
    constexpr bool IsMatrix<T&> = IsMatrix<T>;

    template <typename T>
    constexpr bool IsMatrix<T&&> = IsMatrix<T>;

    /// is batch scalar
    template <typename T>
    constexpr bool IsBatchScalar = false;

    template <typename T>
    constexpr bool IsBatchScalar<const T> = IsBatchScalar<T>;

    template <typename T>
    constexpr bool IsBatchScalar<T&> = IsBatchScalar<T>;

    template <typename T>
    constexpr bool IsBatchScalar<const T&> = IsBatchScalar<T>;

    template <typename T>
    constexpr bool IsBatchScalar<T&&> = IsBatchScalar<T>;

    template <typename T>
    constexpr bool IsBatchScalar<const T&&> = IsBatchScalar<T>;

    /// is batch matrix
    template <typename T>
    constexpr bool IsBatchMatrix = false;

    template <typename T>
    constexpr bool IsBatchMatrix<const T> = IsBatchMatrix<T>;

    template <typename T>
    constexpr bool IsBatchMatrix<T&> = IsBatchMatrix<T>;

    template <typename T>
    constexpr bool IsBatchMatrix<const T&> = IsBatchMatrix<T>;

    template <typename T>
    constexpr bool IsBatchMatrix<T&&> = IsBatchMatrix<T>;

    template <typename T>
    constexpr bool IsBatchMatrix<const T&&> = IsBatchMatrix<T>;

    // 实现的方式很新奇，类似与 switch 语句
    template <typename T>
    struct DataCategory_
    {
    private:
        template <bool isScalar, bool isMatrix, bool isBatchScalar, bool isBatchMatrix, typename TDummy = void>
        struct helper;

        template <typename TDummy>
        struct helper<true, false, false, false, TDummy>
        {
            using type = CategoryTags::Scalar;
        };

        template <typename TDummy>
        struct helper<false, true, false, false, TDummy>
        {
            using type = CategoryTags::Matrix;
        };

        template <typename TDummy>
        struct helper<false, false, true, false, TDummy>
        {
            using type = CategoryTags::BatchScalar;
        };

        template <typename TDummy>
        struct helper<false, false, false, true, TDummy>
        {
            using type = CategoryTags::BatchMatrix;
        };

    public:
        using type = typename helper<IsScalar<T>, IsMatrix<T>, IsBatchScalar<T>, IsBatchMatrix<T>>::type;
    };

    template <typename T>
    using DataCategory = typename DataCategory_<T>::type;

    template <typename T>
    struct IsIterator_
    {
        template <typename R>
        static std::true_type Test(typename std::iterator_traits<R>::iterator_category*);

        template <typename R>
        static std::false_type Test(...);

        static constexpr bool value = decltype(Test<T>(nullptr))::value;
    };

    template <typename T>
    constexpr bool IsIterator = IsIterator_<T>::value;
}
#endif //TRAITS_H
