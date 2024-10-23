//
// Created by wxk on 2024/10/23.
//

#ifndef LOWER_ACCESS_H
#define LOWER_ACCESS_H

namespace MetaNN
{
    /// lower access
    template<typename TData>
    struct LowerAccessImpl;

    template <typename TData>
    auto LowerAccess(TData&& p)
    {
        using RawType = RemConstRef<TData>;
        return LowerAccessImpl<RawType>(std::forward<TData>(p));
    }
}
#endif //LOWER_ACCESS_H
