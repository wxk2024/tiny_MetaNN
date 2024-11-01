//
// Created by wxk on 2024/11/1.
//

#ifndef NEGATIVE_LOG_LIKELIHOOD_DERIVATIVE_H
#define NEGATIVE_LOG_LIKELIHOOD_DERIVATIVE_H
#include <type_traits>
#include <vector>
#include <cmath>

namespace MetaNN {
    // CategoryTags::Scalar : 上层的梯度信息
    // CategoryTags::Matrix : 标注信息
    // CategoryTags::Matrix : 前向传播的结果
    template<>
    struct OperCategory_<TernaryOpTags::NegativeLogLikelihoodDerivative,
                CategoryTags::Scalar, CategoryTags::Matrix, CategoryTags::Matrix> {
        using type = CategoryTags::Matrix;
    };

    template<>
    struct OperCategory_<TernaryOpTags::NegativeLogLikelihoodDerivative,
                CategoryTags::BatchScalar, CategoryTags::BatchMatrix, CategoryTags::BatchMatrix> {
        using type = CategoryTags::BatchMatrix;
    };

    template<>
    class OperOrganizer<TernaryOpTags::NegativeLogLikelihoodDerivative, CategoryTags::Matrix> {
    public:
        template<typename TD1, typename TD2, typename TD3>
        OperOrganizer(const TD1 &data1, const TD2 &data2, const TD3 &data3)
            : m_rowNum(data2.RowNum())
              , m_colNum(data2.ColNum()) {
            assert(data2.RowNum() == data3.RowNum());
            assert(data2.ColNum() == data3.ColNum());
        }

        size_t RowNum() const { return m_rowNum; }
        size_t ColNum() const { return m_colNum; }

    private:
        size_t m_rowNum;
        size_t m_colNum;
    };

    template<>
    class OperOrganizer<TernaryOpTags::NegativeLogLikelihoodDerivative, CategoryTags::BatchMatrix> {
    public:
        template<typename TD1, typename TD2, typename TD3>
        OperOrganizer(const TD1 &data1, const TD2 &data2, const TD3 &data3)
            : m_rowNum(data2.RowNum())
              , m_colNum(data2.ColNum())
              , m_batchNum(data2.BatchNum()) {
            assert(data2.RowNum() == data3.RowNum());
            assert(data2.ColNum() == data3.ColNum());
            assert(data2.BatchNum() == data3.BatchNum());
        }

        size_t RowNum() const { return m_rowNum; }
        size_t ColNum() const { return m_colNum; }
        size_t BatchNum() const { return m_batchNum; }

    private:
        size_t m_rowNum;
        size_t m_colNum;
        size_t m_batchNum;
    };

    template<typename TOp1, typename TOp2, typename TOp3>
    struct OperElementType_<TernaryOpTags::NegativeLogLikelihoodDerivative,
                TOp1, TOp2, TOp3> {
        using type = typename TOp2::ElementType;
    };

    template<typename TOp1, typename TOp2, typename TOp3>
    struct OperDeviceType_<TernaryOpTags::NegativeLogLikelihoodDerivative,
                TOp1, TOp2, TOp3> {
        using type = typename TOp2::DeviceType;
    };
}
#endif //NEGATIVE_LOG_LIKELIHOOD_DERIVATIVE_H
