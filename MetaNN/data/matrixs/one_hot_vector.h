//
// Created by wxk on 2024/10/24.
//

#ifndef ONE_HOT_VECTOR_H
#define ONE_HOT_VECTOR_H

namespace MetaNN {
    namespace NSOneHotVector {
    }
    // 独热向量在这里实现的时候是 行向量！
    template<typename TElem, typename TDevice>
    class OneHotVector {
    public:
        using ElementType = TElem;
        using DeviceType = TDevice;

    public:
        OneHotVector(size_t p_colNum, size_t p_hotPos)
            : m_colNum(p_colNum)
              , m_hotPos(p_hotPos) {
            assert(p_hotPos < m_colNum);
        }

        bool operator==(const OneHotVector &val) const {
            return (m_hotPos == val.m_hotPos) &&
                   (m_colNum == val.m_colNum);
        }

        template<typename TOtherType>
        bool operator==(const TOtherType &) const {
            return false;
        }

        template<typename TData>
        bool operator!=(const TData &val) const {
            return !(operator==(val));
        }

        // 只有一行
        size_t RowNum() const { return 1; }
        size_t ColNum() const { return m_colNum; }

        auto EvalRegister() const {
            using TEvalUnit = NSOneHotVector::EvalUnit<ElementType, DeviceType>;
            using TEvalGroup = TrivalEvalGroup<TEvalUnit>;
            if (!m_evalBuf.IsEvaluated()) {
                auto evalHandle = m_evalBuf.Handle();
                decltype(auto) outputPtr = evalHandle.DataPtr();
                TEvalUnit unit(std::move(evalHandle), 1, m_colNum, m_hotPos);
                EvalPlan<DeviceType>::template Register<TEvalGroup>(std::move(unit), outputPtr, {});
            }
            return m_evalBuf.ConstHandle();
        }

        auto HotPos() const {
            return m_hotPos;
        }

    private:
        size_t m_colNum;
        size_t m_hotPos;
        EvalBuffer<Matrix<ElementType, DeviceType> > m_evalBuf;
    };

    template<typename TElem, typename TDevice>
    constexpr bool IsMatrix<OneHotVector<TElem, TDevice> > = true;
}
#endif //ONE_HOT_VECTOR_H
