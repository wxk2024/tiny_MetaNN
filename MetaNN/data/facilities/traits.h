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
}
#endif //TRAITS_H
