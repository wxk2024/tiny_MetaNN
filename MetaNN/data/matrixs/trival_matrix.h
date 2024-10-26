//
// Created by wxk on 2024/10/23.
//

#ifndef TRIVAL_MATRIX_H
#define TRIVAL_MATRIX_H
#include <utility>
namespace MetaNN{
    // 模板参数相较于 matrix 多了一个 TScalar,一个标量除了
    // TElem 和 TDevice 表示对其求值之后结果对应的计算单元和计算设备
    // 我们允许 TrivalMatrix 与其包含的标量在计算单元与计算设备上存在一定程度上的差异只要满足
    // 1.标量的计算单元类型可以被隐式转换成 TrivalMatrix 的计算单元类型。
    // 2.标量的计算设备是 CPU 或者与 TrivalMatrix 的计算设备类型相同。
    template<typename TElem, typename TDevice, typename TScalar>
    class TrivalMatrix
    {
    public:
        using ElementType = TElem;
        using DeviceType = TDevice;

    public:
        TrivalMatrix(std::size_t p_rowNum, size_t p_colNum,
                     TScalar p_val)
            : m_val(p_val)
            , m_rowNum(p_rowNum)
            , m_colNum(p_colNum) {}

        bool operator== (const TrivalMatrix& val) const
        {
            return (m_val == val.m_val) &&
                   (m_rowNum == val.m_rowNum) &&
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

        size_t RowNum() const
        {
            return m_rowNum;
        }

        size_t ColNum() const
        {
            return m_colNum;
        }

        auto EvalRegister() const
        {
            using TEvalUnit = NSTrivalMatrix::EvalUnit<ElementType, DeviceType>;
            using TEvalGroup = TrivalEvalGroup<TEvalUnit>;
            if (!m_evalBuf.IsEvaluated())
            {
                auto evalHandle = m_evalBuf.Handle();
                const void* outputPtr = evalHandle.DataPtr();
                TEvalUnit unit(std::move(evalHandle), m_rowNum, m_colNum, m_val);
                EvalPlan<DeviceType>::template Register<TEvalGroup>(std::move(unit), outputPtr, {});
            }
            return m_evalBuf.ConstHandle();
        }

        auto ElementValue() const
        {
            return m_val;
        }

    private:
        TScalar m_val;
        size_t m_rowNum;
        size_t m_colNum;
        EvalBuffer<Matrix<ElementType, DeviceType>> m_evalBuf;  //用户保存求值后的结果
    };

    template <typename TElem, typename TDevice, typename TScalar>
    constexpr bool IsMatrix<TrivalMatrix<TElem, TDevice, TScalar>> = true;

    //进行了检查，而后创建 trivalmatrix
    template<typename TElem, typename TDevice, typename TVal>
    auto MakeTrivalMatrix(size_t rowNum, size_t colNum, TVal&& m_val)
    {
        using RawVal = RemConstRef<TVal>;

        if constexpr (IsScalar<RawVal>)
        {
            static_assert(std::is_same<typename RawVal::DeviceType, TDevice>::value ||
                          std::is_same<typename RawVal::DeviceType, DeviceTags::CPU>::value);
            return TrivalMatrix<TElem, TDevice, RawVal>(rowNum, colNum, std::forward<TVal>(m_val));
        }
        else
        {
            TElem tmpElem = static_cast<TElem>(m_val);
            Scalar<TElem, DeviceTags::CPU> scalar(std::move(tmpElem));
            return TrivalMatrix<TElem, TDevice, Scalar<TElem, DeviceTags::CPU>>(rowNum, colNum, std::move(scalar));
        }
    }
}
#endif //TRIVAL_MATRIX_H
