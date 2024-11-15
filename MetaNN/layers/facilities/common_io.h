//
// Created by wxk on 2024/11/6.
//

#ifndef COMMON_IO_H
#define COMMON_IO_H
#include <MetaNN/facilities/var_type_dict.h>

namespace MetaNN
{
    // 正向传播时：层的输入容器
    struct LayerIO : public VarTypeDict<LayerIO> {};

    // 反向传播时：最后一层的输入容器
    struct CostLayerIn : public VarTypeDict<CostLayerIn, struct CostLayerLabel> {};

    struct RnnLayerHiddenBefore;
}
#endif //COMMON_IO_H
