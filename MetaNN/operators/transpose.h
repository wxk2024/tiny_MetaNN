//
// Created by wxk on 2024/10/30.
//

#ifndef TRANSPOSE_H
#define TRANSPOSE_H

namespace MetaNN {
    // 需要对运算模板的父类 OperOrganizer 进行特化，得到尺寸
    template<>
    class OperOrganizer<UnaryOpTags::Transpose, CategoryTags::Matrix> {
    public:
        template<typename TData>
        OperOrganizer(const TData &data)
            : m_rowNum(data.ColNum())
              , m_colNum(data.RowNum()) {
        }

        size_t RowNum() const { return m_rowNum; }
        size_t ColNum() const { return m_colNum; }

    private:
        size_t m_rowNum;
        size_t m_colNum;
    };

    template<>
    class OperOrganizer<UnaryOpTags::Transpose, CategoryTags::BatchMatrix>
            : public OperOrganizer<UnaryOpTags::Transpose, CategoryTags::Matrix> {
        using BaseType = OperOrganizer<UnaryOpTags::Transpose, CategoryTags::Matrix>;

    public:
        template<typename TData>
        OperOrganizer(const TData &data)
            : BaseType(data)
              , m_batchNum(data.BatchNum()) {
        }

        size_t BatchNum() const { return m_batchNum; }

    private:
        size_t m_batchNum;
    };

    template<typename TP>
    struct OperTranspose_ {
        // valid check
    private:
        using rawM = RemConstRef<TP>;

    public:
        static constexpr bool valid = IsMatrix<rawM> || IsBatchMatrix<rawM>;

    public:
        static auto Eval(TP &&p_m) {
            using ResType = UnaryOp<UnaryOpTags::Transpose, rawM>;
            return ResType(std::forward<TP>(p_m));
        }
    };

    template<typename TP,
        std::enable_if_t<OperTranspose_<TP>::valid>* = nullptr>
    auto Transpose(TP &&p_m) {
        return OperTranspose_<TP>::Eval(std::forward<TP>(p_m));
    }
}
#endif //TRANSPOSE_H
