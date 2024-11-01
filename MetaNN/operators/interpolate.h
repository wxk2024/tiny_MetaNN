//
// Created by wxk on 2024/11/1.
//

#ifndef INTERPOLATE_H
#define INTERPOLATE_H

namespace MetaNN {
    template<typename TP1, typename TP2, typename TP3>
    struct OperInterpolate_ {
        // valid check
    private:
        using rawM1 = RemConstRef<TP1>;
        using rawM2 = RemConstRef<TP2>;
        using rawM3 = RemConstRef<TP3>;

    public:
        static constexpr bool valid = (IsMatrix<rawM1> && IsMatrix<rawM2> && IsMatrix<rawM3>) ||
                                      (IsBatchMatrix<rawM1> && IsBatchMatrix<rawM2> && IsBatchMatrix<rawM3>);

    public:
        template<typename T1, typename T2, typename T3,
            std::enable_if_t<std::is_same<T1, T2>::value>* = nullptr,
            std::enable_if_t<std::is_same<T2, T3>::value>* = nullptr>
        static auto Eval(TP1 &&p_m1, TP2 &&p_m2, TP3 &&p_m3) {
            static_assert(std::is_same<typename rawM1::ElementType, typename rawM2::ElementType>::value,
                          "Matrices with different element types cannot interpolate directly");
            static_assert(std::is_same<typename rawM1::DeviceType, typename rawM2::DeviceType>::value,
                          "Matrices with different device types cannot interpolate directly");

            static_assert(std::is_same<typename rawM1::ElementType, typename rawM3::ElementType>::value,
                          "Matrices with different element types cannot interpolate directly");
            static_assert(std::is_same<typename rawM1::DeviceType, typename rawM3::DeviceType>::value,
                          "Matrices with different device types cannot interpolate directly");

            using ResType = TernaryOp<TernaryOpTags::Interpolate, rawM1, rawM2, rawM3>;
            return ResType(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2), std::forward<TP3>(p_m3));
        }
    };

    template<typename TP1, typename TP2, typename TP3,
        std::enable_if_t<OperInterpolate_<TP1, TP2, TP3>::valid>* = nullptr>
    auto Interpolate(TP1 &&p_m1, TP2 &&p_m2, TP3 &&p_lambda) {
        return OperInterpolate_<TP1, TP2, TP3>::
                template Eval<DataCategory<TP1>, DataCategory<TP2>, DataCategory<TP3> >(std::forward<TP1>(p_m1),
                    std::forward<TP2>(p_m2),
                    std::forward<TP3>(p_lambda));
    }
}
#endif //INTERPOLATE_H
