//
// Created by wxk on 2024/10/23.
//
#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/data/matrixs/matrix.h>
#include <cassert>
#include <stdexcept>

#ifndef ZERO_MATRIX_H
#define ZERO_MATRIX_H
namespace MetaNN{
    template <typename TElem, typename TDevice>
    class ZeroMatrix
    {
    public:
        using ElementType = TElem;
        using DeviceType = TDevice;

    public:
        ZeroMatrix(size_t p_rowNum, size_t p_colNum)
            : m_rowNum(p_rowNum)
            , m_colNum(p_colNum) {}

        bool operator== (const ZeroMatrix& val) const
        {
            return (m_rowNum == val.m_rowNum) &&
                   (m_colNum == val.m_colNum);
        }

        template <typename TOtherType>
        bool operator== (const TOtherType&) const
        {
            return false;
        }

        template <typename TData>
        bool operator!= (const TData& val) const
        {
            return !(operator==(val));
        }

        size_t RowNum() const { return m_rowNum; }

        size_t ColNum() const { return m_colNum; }

        auto EvalRegister() const
        {
            using TEvalUnit = NSZeroMatrix::EvalUnit<ElementType, DeviceType>;
            using TEvalGroup = TrivalEvalGroup<TEvalUnit>;
            if (!m_evalBuf.IsEvaluated())
            {
                auto evalHandle = m_evalBuf.Handle();
                decltype(auto) outPtr = evalHandle.DataPtr();
                TEvalUnit unit(std::move(evalHandle), m_rowNum, m_colNum);
                EvalPlan<DeviceType>::template Register<TEvalGroup>(std::move(unit), outPtr, {});
            }
            return m_evalBuf.ConstHandle();
        }

    private:
        // 只包含了行数和列数，大大减少了存储空间
        size_t m_rowNum;
        size_t m_colNum;
        EvalBuffer<Matrix<ElementType, DeviceType>> m_evalBuf;
    };

    template <typename TElem, typename TDevice>
    constexpr bool IsMatrix<ZeroMatrix<TElem, TDevice>> = true;
}
#endif //ZERO_MATRIX_H
