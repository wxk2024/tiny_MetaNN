//
// Created by wxk on 2024/10/22.
//

#ifndef EVAL_HANDLE_H
#define EVAL_HANDLE_H
#include <utility>
template <typename TData>
class ConstEvalHandle
{
public:
    ConstEvalHandle(TData data)
        : m_constData(std::move(data))
    {}

    const TData& Data() const
    {
        return m_constData;
    }

    const void* DataPtr() const
    {
        return &m_constData;
    }

private:
    TData m_constData;
};

namespace MetaNN {
    template <typename TData>
    auto MakeConstEvalHandle(const TData& data)
    {
        return ConstEvalHandle<TData>(data);
    }
}


#endif //EVAL_HANDLE_H
