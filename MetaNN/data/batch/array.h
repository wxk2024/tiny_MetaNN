//
// Created by wxk on 2024/10/24.
//
#ifndef ARRAY_H
#define ARRAY_H
#include <cassert>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/data/batch/batch.h>

namespace MetaNN {
    template<typename TData, typename TDataCate>
    class ArrayImp;

    template<typename TData>
    class Array : public ArrayImp<TData, DataCategory<TData> > {
    public:
        using ElementType = typename TData::ElementType;
        using DeviceType = typename TData::DeviceType;
        using ArrayImp<TData, DataCategory<TData> >::ArrayImp;
    };

    template<typename TData>
    constexpr bool IsBatchMatrix<Array<TData> > = IsMatrix<TData>;

    template<typename TData>
    constexpr bool IsBatchScalar<Array<TData> > = IsScalar<TData>;

    namespace NSArray {
    }

    // 针对 Matrix 的特化
    template<typename TData>
    class ArrayImp<TData, CategoryTags::Matrix> {
    public:
        using ElementType = typename TData::ElementType;
        using DeviceType = typename TData::DeviceType;

        // 构造函数(1/2)
        ArrayImp(size_t rowNum = 0, size_t colNum = 0)
            : m_rowNum(rowNum)
              , m_colNum(colNum)
              , m_buffer(new std::vector<TData>()) {
        }

        // 构造函数(2/2)
        template<typename TIterator, std::enable_if_t<IsIterator<TIterator> >* = nullptr>
        ArrayImp(TIterator b, TIterator e)
            : m_rowNum(0)
              , m_colNum(0)
              , m_buffer(new std::vector<TData>(b, e)) {
            const auto &buffer = *m_buffer;
            if (!buffer.empty()) {
                m_rowNum = buffer[0].RowNum();
                m_colNum = buffer[0].ColNum();
                // 验证是否行列一致，且和函数参数一致
                for (size_t i = 1; i < buffer.size(); ++i) {
                    if ((buffer[i].RowNum() != m_rowNum) ||
                        (buffer[i].ColNum() != m_colNum)) {
                        throw std::runtime_error("Dimension mismatch");
                    }
                }
            }
        }

    public:
        size_t RowNum() const { return m_rowNum; }
        size_t ColNum() const { return m_colNum; }

        size_t BatchNum() const {
            return m_buffer->size();
        }
        // 与 STL 兼容的一些接口
        size_t size() const { return m_buffer->size(); }
        void push_back(TData mat) {
            assert(AvailableForWrite());
            if ((mat.RowNum() != m_rowNum) || (mat.ColNum() != m_colNum)) {
                throw std::runtime_error("Dimension mismatch");
            }
            m_buffer->emplace_back(std::move(mat));
        }

        template<typename... TArgs>
        void emplace_back(TArgs &&... args) {
            assert(AvailableForWrite()); // 我们不希望有多个副本的时候可以写入
            TData tmp(std::forward<TArgs>(args)...);
            if ((tmp.RowNum() != m_rowNum) || (tmp.ColNum() != m_colNum)) {
                throw std::runtime_error("Dimension mismatch");
            }
            m_buffer.emplace_back(std::move(tmp));
        }

        void reserve(size_t num) {
            assert(AvailableForWrite());
            m_buffer.reserve(num);
        }

        void clear() {
            assert(AvailableForWrite());
            m_buffer.clear();
        }

        bool empty() const {
            return m_buffer->empty();
        }

        const auto &operator[](size_t id) const {
            return (*m_buffer)[id];
        }

        auto &operator[](size_t id) {
            return (*m_buffer)[id];
        }

        auto begin() { return m_buffer->begin(); }
        auto begin() const { return m_buffer->begin(); }
        auto end() { return m_buffer->end(); }
        auto end() const { return m_buffer->end(); }

        bool operator==(const Array<TData> &val) const {
            const ArrayImp<TData, CategoryTags::Matrix> &tmp = static_cast<const ArrayImp<TData, CategoryTags::Matrix>
                &>(val);
            return m_buffer == tmp.m_buffer;
        }

        template<typename TOtherType>
        bool operator==(const TOtherType &) const {
            return false;
        }

        template<typename TCompData>
        bool operator!=(const TCompData &val) const {
            return !(operator==(val));
        }

        auto EvalRegister() const {
            if (!m_evalBuf.IsEvaluated()) {
                using TOpEvalHandle = std::decay_t<decltype(std::declval<TData>().EvalRegister())>;
                std::vector<TOpEvalHandle> handleBuf;
                std::vector<const void *> depVec;
                handleBuf.reserve(this->size());
                depVec.reserve(this->size());
                for (size_t i = 0; i < this->size(); ++i) {
                    handleBuf.push_back((*this)[i].EvalRegister());
                    depVec.push_back(handleBuf.back().DataPtr());
                }

                auto outHandle = m_evalBuf.Handle();

                using EvalUnit = NSArray::EvalUnit<TOpEvalHandle, ElementType, DeviceType, CategoryTags::Matrix>;
                using GroupType = TrivalEvalGroup<EvalUnit>;

                const void *dataPtr = outHandle.DataPtr();
                EvalUnit unit(std::move(handleBuf), std::move(outHandle));
                EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr, std::move(depVec));
            }
            return m_evalBuf.ConstHandle();
        }

        bool AvailableForWrite() const {
            // 涉及了求值的一部分逻辑
            // 如果求值操作进行过，那么 isEvaluated 返回为真，此时不应该继续 push_back 写操作
            return (!m_evalBuf.IsEvaluated()) && (m_buffer.use_count() == 1);
        }

    protected:
        size_t m_rowNum;
        size_t m_colNum;
        std::shared_ptr<std::vector<TData> > m_buffer;
        // EvalBuffer 包含了一个内部状态，表示之前是否完成过求值，对于一般的数据对象，如果它在之前已经求过值了，那么
        // 下次再对其求值，就直接返回 EvalBuffer 所存储的求值结果
        EvalBuffer<Batch<ElementType, DeviceType, CategoryTags::Matrix> > m_evalBuf; // 用来存储求值的结果
    };

    // 针对 Scalar 标量的特化
    template<typename TData>
    class ArrayImp<TData, CategoryTags::Scalar> {
    public:
        using ElementType = typename TData::ElementType;
        using DeviceType = typename TData::DeviceType;

        ArrayImp(size_t rowNum = 0, size_t colNum = 0)
            : m_buffer(new std::vector<TData>()) {
        }

        template<typename TIterator, std::enable_if_t<IsIterator<TIterator> >* = nullptr>
        ArrayImp(TIterator b, TIterator e)
            : m_buffer(new std::vector<TData>(b, e)) {
        }

    public:
        size_t BatchNum() const {
            return m_buffer->size();
        }

        size_t size() const { return m_buffer->size(); }


        void push_back(TData mat) {
            assert(AvailableForWrite());
            m_buffer->emplace_back(std::move(mat));
        }

        template<typename... TArgs>
        void emplace_back(TArgs &&... args) {
            assert(AvailableForWrite());
            TData tmp(std::forward<TArgs>(args)...);
            m_buffer.emplace_back(std::move(tmp));
        }

        void reserve(size_t num) {
            assert(AvailableForWrite());
            m_buffer.reserve(num);
        }

        void clear() {
            assert(AvailableForWrite());
            m_buffer.clear();
        }

        bool empty() const {
            return m_buffer->empty();
        }

        const auto &operator[](size_t id) const {
            return (*m_buffer)[id];
        }

        auto &operator[](size_t id) {
            return (*m_buffer)[id];
        }

        auto begin() { return m_buffer->begin(); }
        auto begin() const { return m_buffer->begin(); }
        auto end() { return m_buffer->end(); }
        auto end() const { return m_buffer->end(); }

        bool operator==(const Array<TData> &val) const {
            const ArrayImp<TData, CategoryTags::Scalar> &tmp = static_cast<const ArrayImp<TData, CategoryTags::Scalar>
                &>(val);
            return m_buffer == tmp.m_buffer;
        }

        template<typename TOtherType>
        bool operator==(const TOtherType &) const {
            return false;
        }

        template<typename TCompData>
        bool operator!=(const TCompData &val) const {
            return !(operator==(val));
        }

        auto EvalRegister() const {
            if (!m_evalBuf.IsEvaluated()) {
                using TOpEvalHandle = std::decay_t<decltype(std::declval<TData>().EvalRegister())>;
                std::vector<TOpEvalHandle> handleBuf;
                std::vector<const void *> depVec;
                handleBuf.reserve(this->size());
                depVec.reserve(this->size());
                for (size_t i = 0; i < this->size(); ++i) {
                    handleBuf.push_back((*this)[i].EvalRegister());
                    depVec.push_back(handleBuf.back().DataPtr());
                }

                auto outHandle = m_evalBuf.Handle();

                using EvalUnit = NSArray::EvalUnit<TOpEvalHandle, ElementType, DeviceType, CategoryTags::Scalar>;
                using GroupType = TrivalEvalGroup<EvalUnit>;

                const void *dataPtr = outHandle.DataPtr();
                EvalUnit unit(std::move(handleBuf), std::move(outHandle));
                EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr, std::move(depVec));
            }
            return m_evalBuf.ConstHandle();
        }

        bool AvailableForWrite() const {
            return (!m_evalBuf.IsEvaluated()) && (m_buffer.use_count() == 1);
        }

    protected:
        std::shared_ptr<std::vector<TData> > m_buffer;
        EvalBuffer<Batch<ElementType, DeviceType, CategoryTags::Scalar> > m_evalBuf;
    };

    template<typename TIterator>
    auto MakeArray(TIterator beg, TIterator end) {
        using TData = typename std::iterator_traits<TIterator>::value_type;
        using RawData = RemConstRef<TData>;

        return Array<RawData>(beg, end);
    }
}
#endif //ARRAY_H
