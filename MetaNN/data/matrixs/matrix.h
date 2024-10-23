//
// Created by wxk on 2024/10/22.
//

#ifndef MATRIX_H
#define MATRIX_H

namespace MetaNN{
    // matrices:计算类型，计算设备
    template<typename TElement, typename TDevice>
    class Matrix;

    template <typename TElement, typename TDevice>
    constexpr bool IsMatrix<Matrix<TElement, TDevice>> = true;
}
#endif //MATRIX_H
