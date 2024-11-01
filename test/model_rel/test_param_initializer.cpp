//
// Created by wxk on 2024/11/1.
//
#include <catch2/catch_test_macros.hpp>
#include <MetaNN/policies/policy_container.h>
#include <MetaNN/facilities/traits.h>
#include <MetaNN/policies/policy_selector.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/policies/change_policy.h>
#include <MetaNN/layers/facilities/policies.h>
#include <cmath>
#include <set>
#include <type_traits>
#include <MetaNN/meta_nn.h>
#include <iostream>
#include <MetaNN/model_rel/param_initializer/constant_filler.h>
#include <MetaNN/data/matrixs/cpu_matrix.h>
#include <MetaNN/model_rel/param_initializer/gaussian_filler.h>
#include <MetaNN/model_rel/param_initializer/var_scale_filler.h>
#include <MetaNN/model_rel/param_initializer/facilities/policies.h>

using namespace std;
using namespace MetaNN;
TEST_CASE("constant filler", "[model_rel param initializer]") {
    ConstantFiller filler(0);
    Matrix<float, DeviceTags::CPU> mat (11, 13);
    filler.Fill(mat, 11, 13);
    for (size_t i = 0; i < 11; ++i)
    {
        for (size_t j = 0; j < 13; ++j)
        {
            assert(fabs(mat(i, j)) < 0.0001);
        }
    }

    ConstantFiller filler2(1.5f);
    Matrix<float, DeviceTags::CPU> mat2 (21, 33);
    filler2.Fill(mat2, 21, 33);
    for (size_t i = 0; i < 21; ++i)
    {
        for (size_t j = 0; j < 33; ++j)
        {
            assert(fabs(mat2(i, j) - 1.5) < 0.0001);
        }
    }

    cout << "done" << endl;
}
TEST_CASE("gaussian filler", "[model_rel param initializer]") {
    cout << "test gaussian filler case 1 ...";

    GaussianFiller filler(1.5, 3.3);
    Matrix<float, DeviceTags::CPU> mat (1000, 3000);
    filler.Fill(mat, 1000, 3000);

    float mean = 0;
    for (size_t i = 0; i < mat.RowNum(); ++i)
    {
        for (size_t j = 0; j < mat.ColNum(); ++j)
        {
            mean += mat(i, j);
        }
    }
    mean /= mat.RowNum() * mat.ColNum();

    float var = 0;
    for (size_t i = 0; i < mat.RowNum(); ++i)
    {
        for (size_t j = 0; j < mat.ColNum(); ++j)
        {
            var += (mat(i, j) - mean) * (mat(i, j) - mean);
        }
    }
    var /= mat.RowNum() * mat.ColNum();

    // mean = 1.5, std = 3.3
    cout << "mean-delta = " << fabs(mean-1.5) << " std-delta = " << fabs(sqrt(var)-3.3) << ' ';
    cout << "done" << endl;
}

TEST_CASE("xavier filler", "[model_rel param initializer]") {
    cout << "test xavier filler case 1 ...";

    XavierFiller<PolicyContainer<PUniformVarScale>> filler;
    Matrix<float, DeviceTags::CPU> mat (100, 200);
    filler.Fill(mat, 100, 200);

    float mean = 0;
    for (size_t i = 0; i < mat.RowNum(); ++i)
    {
        for (size_t j = 0; j < mat.ColNum(); ++j)
        {
            mean += mat(i, j);
        }
    }
    mean /= mat.RowNum() * mat.ColNum();

    float var = 0;
    for (size_t i = 0; i < mat.RowNum(); ++i)
    {
        for (size_t j = 0; j < mat.ColNum(); ++j)
        {
            var += (mat(i, j) - mean) * (mat(i, j) - mean);
        }
    }
    var /= mat.RowNum() * mat.ColNum();

    // std = 0.0816 (sqrt(1/150)) = (2/(100 + 200))
    cout << "mean-delta = " << fabs(mean) << " std-delta = " << fabs(sqrt(var)-0.0816) << ' ';
    cout << "done" << endl;
}

TEST_CASE("xavier filler 2", "[model_rel param initializer]") {
    cout << "test xavier filler case 2 ...";

    XavierFiller<PolicyContainer<PNormVarScale /*策略无效的*/>> filler;
    Matrix<float, DeviceTags::CPU> mat (100, 200);
    filler.Fill(mat, 100, 200);

    float mean = 0;
    for (size_t i = 0; i < mat.RowNum(); ++i)
    {
        for (size_t j = 0; j < mat.ColNum(); ++j)
        {
            mean += mat(i, j);
        }
    }
    mean /= mat.RowNum() * mat.ColNum();

    float var = 0;
    for (size_t i = 0; i < mat.RowNum(); ++i)
    {
        for (size_t j = 0; j < mat.ColNum(); ++j)
        {
            var += (mat(i, j) - mean) * (mat(i, j) - mean);
        }
    }
    var /= mat.RowNum() * mat.ColNum();

    // std = 0.0816 (sqrt(1/150))
    cout << "mean-delta = " << fabs(mean) << " std = " << fabs(sqrt(var)-0.0816) << ' ';
    cout << "done" << endl;
}

TEST_CASE("msra filler", "[model_rel param initializer]") {
    cout << "test msra filler case 1 ...";

    MSRAFiller<> filler;
    Matrix<float, DeviceTags::CPU> mat (100, 200);
    filler.Fill(mat, 100, 200);

    float mean = 0;
    for (size_t i = 0; i < mat.RowNum(); ++i)
    {
        for (size_t j = 0; j < mat.ColNum(); ++j)
        {
            mean += mat(i, j);
        }
    }
    mean /= mat.RowNum() * mat.ColNum();

    float var = 0;
    for (size_t i = 0; i < mat.RowNum(); ++i)
    {
        for (size_t j = 0; j < mat.ColNum(); ++j)
        {
            var += (mat(i, j) - mean) * (mat(i, j) - mean);
        }
    }
    var /= mat.RowNum() * mat.ColNum();

    // std = 0.1414 (sqrt(2/50))
    cout << "mean-delta = " << fabs(mean) << " std-delta = " << fabs(sqrt(var)-0.1414) << ' ';
    cout << "done" << endl;
}

