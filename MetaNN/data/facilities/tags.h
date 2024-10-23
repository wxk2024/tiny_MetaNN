//
// Created by wxk on 2024/10/20.
//

#ifndef TAGS_H
#define TAGS_H
namespace MetaNN
{
    /// data types
    struct CategoryTags
    {
        struct Scalar;
        struct Matrix;
        struct BatchScalar;
        struct BatchMatrix;
    };
    /// device types
    struct DeviceTags
    {
        struct CPU;
    };
}
#endif //TAGS_H
