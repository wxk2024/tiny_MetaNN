//
// Created by wxk on 2024/11/1.
//

#ifndef DOT_H
#define DOT_H

namespace MetaNN {
    // 矩阵乘的尺寸特化
    template<>
    class OperOrganizer<BinaryOpTags::Dot, CategoryTags::Matrix> {
    public:
        template<typename TD1, typename TD2>
        OperOrganizer(const TD1 &data1, const TD2 &data2)
            : m_rowNum(data1.RowNum())
              , m_colNum(data2.ColNum()) {
            assert(data1.ColNum() == data2.RowNum());
        }

        size_t RowNum() const { return m_rowNum; }
        size_t ColNum() const { return m_colNum; }

    private:
        size_t m_rowNum;
        size_t m_colNum;
    };

    template<>
    class OperOrganizer<BinaryOpTags::Dot, CategoryTags::BatchMatrix> {
    public:
        template<typename TD1, typename TD2>
        OperOrganizer(const TD1 &data1, const TD2 &data2)
            : m_rowNum(data1.RowNum())
              , m_colNum(data2.ColNum())
              , m_batchNum(data1.BatchNum()) {
            assert(data1.ColNum() == data2.RowNum());
            assert(data1.BatchNum() == data2.BatchNum());
        }

        size_t RowNum() const { return m_rowNum; }
        size_t ColNum() const { return m_colNum; }
        size_t BatchNum() const { return m_batchNum; }

    private:
        size_t m_rowNum;
        size_t m_colNum;
        size_t m_batchNum;
    };

    template<typename TP1, typename TP2>
    struct OperDot_ {
        // valid check
    private:
        using rawM1 = RemConstRef<TP1>;
        using rawM2 = RemConstRef<TP2>;

    public:
        static constexpr bool valid = (IsMatrix<rawM1> && IsMatrix<rawM2>) ||
                                      (IsBatchMatrix<rawM1> && IsMatrix<rawM2>) ||
                                      (IsMatrix<rawM1> && IsBatchMatrix<rawM2>) ||
                                      (IsBatchMatrix<rawM1> && IsBatchMatrix<rawM2>);

    public:
        template<typename T1, typename T2,
            std::enable_if_t<std::is_same<T1, T2>::value>* = nullptr>
        static auto Eval(TP1 &&p_m1, TP2 &&p_m2) {
            static_assert(std::is_same<typename rawM1::ElementType, typename rawM2::ElementType>::value,
                          "Matrices with different element types cannot dot directly");
            static_assert(std::is_same<typename rawM1::DeviceType, typename rawM2::DeviceType>::value,
                          "Matrices with different device types cannot dot directly");

            using ResType = BinaryOp<BinaryOpTags::Dot, rawM1, rawM2>;
            return ResType(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));
        }

        template<typename T1, typename T2,
            std::enable_if_t<std::is_same<CategoryTags::BatchMatrix, T1>::value>* = nullptr,
            std::enable_if_t<std::is_same<CategoryTags::Matrix, T2>::value>* = nullptr>
        static auto Eval(TP1 &&p_m1, TP2 &&p_m2) {
            static_assert(std::is_same<typename rawM1::ElementType, typename rawM2::ElementType>::value,
                          "Matrices with different element types cannot dot directly");
            static_assert(std::is_same<typename rawM1::DeviceType, typename rawM2::DeviceType>::value,
                          "Matrices with different device types cannot dot directly");
            // 创建一个临时的 batch 来进行矩阵乘的运算
            Duplicate<rawM2> tmp(std::forward<TP2>(p_m2), p_m1.BatchNum());
            using ResType = BinaryOp<BinaryOpTags::Dot, rawM1, Duplicate<rawM2> >;
            return ResType(std::forward<TP1>(p_m1), std::move(tmp));
        }

        template<typename T1, typename T2,
            std::enable_if_t<std::is_same<CategoryTags::Matrix, T1>::value>* = nullptr,
            std::enable_if_t<std::is_same<CategoryTags::BatchMatrix, T2>::value>* = nullptr>
        static auto Eval(TP1 &&p_m1, TP2 &&p_m2) {
            static_assert(std::is_same<typename rawM1::ElementType, typename rawM2::ElementType>::value,
                          "Matrices with different element types cannot dot directly");
            static_assert(std::is_same<typename rawM1::DeviceType, typename rawM2::DeviceType>::value,
                          "Matrices with different device types cannot dot directly");

            Duplicate<rawM1> tmp(std::forward<TP1>(p_m1), p_m2.BatchNum());
            using ResType = BinaryOp<BinaryOpTags::Dot, Duplicate<rawM1>, rawM2>;
            return ResType(std::move(tmp), std::forward<TP2>(p_m2));
        }
    };

    template<typename TP1, typename TP2,
        std::enable_if_t<OperDot_<TP1, TP2>::valid>* = nullptr>
    auto Dot(TP1 &&p_m1, TP2 &&p_m2) {
        return OperDot_<TP1, TP2>::
                template Eval<DataCategory<TP1>, DataCategory<TP2> >(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));
    }
}
#endif //DOT_H
