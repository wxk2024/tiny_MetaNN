//
// Created by wxk on 2024/10/26.
//

#ifndef TRAITS_H
#define TRAITS_H
namespace MetaNN{
    // 用于指定运算模板的计算单元与计算设备类型。
    // 通常情况下，运算模板具有相同的计算单元和设备，这里就是默认实现
    template <typename TOpTag, typename TOp1, typename...TOperands>
    struct OperElementType_
    {
        using type = typename TOp1::ElementType;
    };

    template <typename TOpTag, typename TOp1, typename...TOperands>
    struct OperDeviceType_
    {
        using type = typename TOp1::DeviceType;
    };
}
#endif //TRAITS_H
