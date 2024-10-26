//
// Created by wxk on 2024/10/26.
//

#ifndef ORGANIZER_H
#define ORGANIZER_H
#include <MetaNN/data/facilities/traits.h>
namespace MetaNN {
    /* 提供尺寸信息：输入是操作和结果类别信息 */
    template <typename TOpTag, typename TCate>
    class OperOrganizer;

    // 针对标量的特化
    template <typename TOpTag>
    class OperOrganizer<TOpTag, CategoryTags::Scalar>
    {
    public:
        // 所有的特化版本中都有一个 模板构造函数
        // 接收运算的输入参数并由此计算出运算结果的尺寸信息
        template <typename THead, typename...TRemain>
        OperOrganizer(const THead&, const TRemain&...)
        {}
    };

    // 针对矩阵的特化
    template <typename TOpTag>
    class OperOrganizer<TOpTag, CategoryTags::Matrix>
    {
    private:
        template <typename THead, typename...TRemain>
        bool SameDim(const THead&, const TRemain&...)
        {
            return true;
        }

        template <typename THead, typename TCur, typename...TRemain>
        bool SameDim(const THead& head, const TCur& cur, const TRemain&...rem)
        {
            const bool tmp = (head.RowNum() == cur.RowNum()) &&
                             (head.ColNum() == cur.ColNum());
            return tmp && SameDim(cur, rem...);
        }

    public:
        template <typename THead, typename...TRemain>
        OperOrganizer(const THead& head, const TRemain&... rem)
            : m_rowNum(head.RowNum())
            , m_colNum(head.ColNum())
        {
            assert(SameDim(head, rem...));
        }

        size_t RowNum() const { return m_rowNum; }
        size_t ColNum() const { return m_colNum; }

    private:
        size_t m_rowNum;
        size_t m_colNum;
    };

    // 针对标量列表的特化
    template <typename TOpTag>
    class OperOrganizer<TOpTag, CategoryTags::BatchScalar>
    {
    private:
        template <typename THead, typename...TRemain>
        bool SameDim(const THead&, const TRemain&...)
        {
            return true;
        }

        template <typename THead, typename TCur, typename...TRemain>
        bool SameDim(const THead& head, const TCur& cur, const TRemain&...rem)
        {
            const bool tmp = (head.BatchNum() == cur.BatchNum());
            return tmp && SameDim(cur, rem...);
        }

    public:
        template <typename THead, typename...TRemain>
        OperOrganizer(const THead& head, const TRemain&... rem)
            : m_batchNum(head.BatchNum())
        {
            // 调用函数进行断言
            assert(SameDim(head, rem...));
        }

        size_t BatchNum() const { return m_batchNum; }

    private:
        size_t m_batchNum;
    };

    //针对 矩阵列表的 特化
    template <typename TOpTag>
    class OperOrganizer<TOpTag, CategoryTags::BatchMatrix>
    {
    private:
        template <typename THead, typename...TRemain>
        bool SameDim(const THead&, const TRemain&...)
        {
            return true;
        }

        template <typename THead, typename TCur, typename...TRemain>
        bool SameDim(const THead& head, const TCur& cur, const TRemain&...rem)
        {
            const bool tmp = (head.RowNum() == cur.RowNum()) &&
                             (head.ColNum() == cur.ColNum()) &&
                             (head.BatchNum() == cur.BatchNum());
            return tmp && SameDim(cur, rem...);
        }

    public:
        template <typename THead, typename...TRemain>
        OperOrganizer(const THead& head, const TRemain&... rem)
            : m_rowNum(head.RowNum())
            , m_colNum(head.ColNum())
            , m_batchNum(head.BatchNum())
        {
            assert(SameDim(head, rem...));
        }

        size_t RowNum() const { return m_rowNum; }
        size_t ColNum() const { return m_colNum; }
        size_t BatchNum() const { return m_batchNum; }

    private:
        size_t m_rowNum;
        size_t m_colNum;
        size_t m_batchNum;
    };
}



#endif //ORGANIZER_H
