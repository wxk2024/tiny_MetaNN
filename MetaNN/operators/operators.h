//
// Created by wxk on 2024/10/28.
//

#ifndef OPERATORS_H
#define OPERATORS_H
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/operators/facilities/category_cal.h>
#include <MetaNN/operators/facilities/oper_seq.h>
#include <MetaNN/operators/facilities/organizer.h>
#include <MetaNN/operators/facilities/traits.h>

namespace MetaNN {
    /*这个才是 “运算模板对象” */
    template<typename TOpTag, typename TData>
    class UnaryOp : public/*操作结果的尺寸*/ OperOrganizer<TOpTag/*操作类型*/, OperCateCal<TOpTag, TData>/*操作结果的数据类型*/> {
        static_assert(std::is_same<RemConstRef<TData>, TData>::value,
                      "TData is not an available type");
        using Cate = OperCateCal<TOpTag, TData>; /*操作结果的数据类型*/

    public:
        // 获取计算单元和设备
        using ElementType = typename OperElementType_<TOpTag, TData>::type; // 浮点数的矩阵或标量，它还是浮点数
        using DeviceType = typename OperDeviceType_<TOpTag, TData>::type;

        UnaryOp(TData data)
            : OperOrganizer<TOpTag, Cate>(data)
              , m_data(std::move(data)) {
        }

        bool operator==(const UnaryOp &val) const {
            return m_data == val.m_data;
        }

        template<typename TOtherData>
        bool operator==(const TOtherData &val) const {
            return false;
        }

        template<typename TOtherData>
        bool operator!=(const TOtherData &val) const {
            return !(operator==(val));
        }

        // 求值
        auto EvalRegister() const {
            if (!m_evalBuf.IsEvaluated()) {
                using TOperSeqCont = typename OperSeq_<TOpTag>::type;

                using THead = SeqHead<TOperSeqCont>;
                using TTail = SeqTail<TOperSeqCont>;
                THead::template EvalRegister<TTail>(m_evalBuf, m_data);
            }
            return m_evalBuf.ConstHandle();
        }

        const TData &Operand() const {
            return m_data;
        }

    private:
        TData m_data;

        using TPrincipal = PrincipalDataType<Cate, ElementType, DeviceType>; // 得到包含计算单元和设备类型的 scalar matrix batch
        EvalBuffer<TPrincipal> m_evalBuf;
    };

    template<typename TOpTag, typename TData1, typename TData2>
    class BinaryOp : public OperOrganizer<TOpTag, OperCateCal<TOpTag, TData1, TData2> > {
        static_assert(std::is_same<RemConstRef<TData1>, TData1>::value,
                      "TData1 is not an available type");
        static_assert(std::is_same<RemConstRef<TData2>, TData2>::value,
                      "TData2 is not an available type");
        using Cate = OperCateCal<TOpTag, TData1, TData2>;

    public:
        using ElementType = typename OperElementType_<TOpTag, TData1, TData2>::type;
        using DeviceType = typename OperDeviceType_<TOpTag, TData1, TData2>::type;

    public:
        BinaryOp(TData1 data1, TData2 data2)
            : OperOrganizer<TOpTag, Cate>(data1, data2)
              , m_data1(std::move(data1))
              , m_data2(std::move(data2)) {
        }

        bool operator==(const BinaryOp &val) const {
            return (m_data1 == val.m_data1) && (m_data2 == val.m_data2);
        }

        template<typename TOtherData>
        bool operator==(const TOtherData &val) const {
            return false;
        }

        template<typename TOtherData>
        bool operator!=(const TOtherData &val) const {
            return !(operator==(val));
        }

        auto EvalRegister() const {
            if (!m_evalBuf.IsEvaluated()) {
                using TOperSeqCont = typename OperSeq_<TOpTag>::type;

                using THead = SeqHead<TOperSeqCont>;
                using TTail = SeqTail<TOperSeqCont>;
                THead::template EvalRegister<TTail>(m_evalBuf, m_data1, m_data2);
            }
            return m_evalBuf.ConstHandle();
        }

        const TData1 &Operand1() const {
            return m_data1;
        }

        const TData2 &Operand2() const {
            return m_data2;
        }

    private:
        TData1 m_data1;
        TData2 m_data2;

        using TPrincipal = PrincipalDataType<Cate, ElementType, DeviceType>;
        EvalBuffer<TPrincipal> m_evalBuf;
    };

    template<typename TOpTag, typename TData1, typename TData2, typename TData3>
    class TernaryOp : public OperOrganizer<TOpTag, OperCateCal<TOpTag, TData1, TData2, TData3> > {
        static_assert(std::is_same<RemConstRef<TData1>, TData1>::value,
                      "TData1 is not an available type");
        static_assert(std::is_same<RemConstRef<TData2>, TData2>::value,
                      "TData2 is not an available type");
        static_assert(std::is_same<RemConstRef<TData3>, TData3>::value,
                      "TData3 is not an available type");
        using Cate = OperCateCal<TOpTag, TData1, TData2, TData3>;

    public:
        using ElementType = typename OperElementType_<TOpTag, TData1, TData2, TData3>::type;
        using DeviceType = typename OperDeviceType_<TOpTag, TData1, TData2, TData3>::type;

    public:
        TernaryOp(TData1 data1, TData2 data2, TData3 data3)
            : OperOrganizer<TOpTag, Cate>(data1, data2, data3)
              , m_data1(std::move(data1))
              , m_data2(std::move(data2))
              , m_data3(std::move(data3)) {
        }

        bool operator==(const TernaryOp &val) const {
            return (m_data1 == val.m_data1) &&
                   (m_data2 == val.m_data2) &&
                   (m_data3 == val.m_data3);
        }

        template<typename TOtherData>
        bool operator==(const TOtherData &val) const {
            return false;
        }

        template<typename TOtherData>
        bool operator!=(const TOtherData &val) const {
            return !(operator==(val));
        }

        auto EvalRegister() const {
            if (!m_evalBuf.IsEvaluated()) {
                using TOperSeqCont = typename OperSeq_<TOpTag>::type;

                using THead = SeqHead<TOperSeqCont>;
                using TTail = SeqTail<TOperSeqCont>;
                THead::template EvalRegister<TTail>(m_evalBuf, m_data1, m_data2, m_data3);
            }
            return m_evalBuf.ConstHandle();
        }

        const TData1 &Operand1() const {
            return m_data1;
        }

        const TData2 &Operand2() const {
            return m_data2;
        }

        const TData3 &Operand3() const {
            return m_data3;
        }

    private:
        TData1 m_data1;
        TData2 m_data2;
        TData3 m_data3;

        using TPrincipal = PrincipalDataType<Cate, ElementType, DeviceType>;
        EvalBuffer<TPrincipal> m_evalBuf;
    };

    // 运算模板运算的结果是什么
    template<typename TOpTag, typename TData>
    constexpr bool IsMatrix<UnaryOp<TOpTag, TData> >
            = std::is_same<OperCateCal<TOpTag, TData>, CategoryTags::Matrix>::value;

    template<typename TOpTag, typename TData>
    constexpr bool IsScalar<UnaryOp<TOpTag, TData> >
            = std::is_same<OperCateCal<TOpTag, TData>, CategoryTags::Scalar>::value;

    template<typename TOpTag, typename TData>
    constexpr bool IsBatchMatrix<UnaryOp<TOpTag, TData> >
            = std::is_same<OperCateCal<TOpTag, TData>, CategoryTags::BatchMatrix>::value;

    template<typename TOpTag, typename TData>
    constexpr bool IsBatchScalar<UnaryOp<TOpTag, TData> >
            = std::is_same<OperCateCal<TOpTag, TData>, CategoryTags::BatchScalar>::value;

    template<typename TOpTag, typename TData1, typename TData2>
    constexpr bool IsScalar<BinaryOp<TOpTag, TData1, TData2> >
            = std::is_same<OperCateCal<TOpTag, TData1, TData2>, CategoryTags::Scalar>::value;

    template<typename TOpTag, typename TData1, typename TData2>
    constexpr bool IsMatrix<BinaryOp<TOpTag, TData1, TData2> >
            = std::is_same<OperCateCal<TOpTag, TData1, TData2>, CategoryTags::Matrix>::value;

    template<typename TOpTag, typename TData1, typename TData2>
    constexpr bool IsBatchMatrix<BinaryOp<TOpTag, TData1, TData2> >
            = std::is_same<OperCateCal<TOpTag, TData1, TData2>, CategoryTags::BatchMatrix>::value;

    template<typename TOpTag, typename TData1, typename TData2>
    constexpr bool IsBatchScalar<BinaryOp<TOpTag, TData1, TData2> >
            = std::is_same<OperCateCal<TOpTag, TData1, TData2>, CategoryTags::BatchScalar>::value;

    template<typename TOpTag, typename TData1, typename TData2, typename TData3>
    constexpr bool IsScalar<TernaryOp<TOpTag, TData1, TData2, TData3> >
            = std::is_same<OperCateCal<TOpTag, TData1, TData2, TData3>, CategoryTags::Scalar>::value;

    template<typename TOpTag, typename TData1, typename TData2, typename TData3>
    constexpr bool IsMatrix<TernaryOp<TOpTag, TData1, TData2, TData3> >
            = std::is_same<OperCateCal<TOpTag, TData1, TData2, TData3>, CategoryTags::Matrix>::value;

    template<typename TOpTag, typename TData1, typename TData2, typename TData3>
    constexpr bool IsBatchMatrix<TernaryOp<TOpTag, TData1, TData2, TData3> >
            = std::is_same<OperCateCal<TOpTag, TData1, TData2, TData3>, CategoryTags::BatchMatrix>::value;

    template<typename TOpTag, typename TData1, typename TData2, typename TData3>
    constexpr bool IsBatchScalar<TernaryOp<TOpTag, TData1, TData2, TData3> >
            = std::is_same<OperCateCal<TOpTag, TData1, TData2, TData3>, CategoryTags::BatchScalar>::value;
}
#endif //OPERATORS_H
