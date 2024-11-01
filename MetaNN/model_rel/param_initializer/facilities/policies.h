//
// Created by wxk on 2024/11/1.
//

#ifndef POLICIES_H
#define POLICIES_H
#include <random>
#include <MetaNN/policies/policy_macro_begin.h>
namespace MetaNN
{
    // 初始化策略组
    struct InitPolicy
    {
        using MajorClass = InitPolicy;

        struct OverallTypeCate;
        struct WeightTypeCate;
        struct BiasTypeCate;

        using Overall = void;
        using Weight = void;
        using Bias = void;


        struct RandEngineTypeCate;
        using RandEngine = std::mt19937;
    };

    TypePolicyTemplate(PInitializerIs,       InitPolicy, Overall);
    TypePolicyTemplate(PWeightInitializerIs, InitPolicy, Weight);
    TypePolicyTemplate(PBiasInitializerIs,   InitPolicy, Bias);
    TypePolicyTemplate(PRandomGeneratorIs,   InitPolicy, RandEngine);

    // 分布策略组
    struct VarScaleFillerPolicy
    {
        using MajorClass = VarScaleFillerPolicy;

        struct DistributeTypeCate
        {
            struct Uniform;
            struct Norm;
        };
        using Distribute = DistributeTypeCate::Uniform;

        struct ScaleModeTypeCate
        {
            struct FanIn;
            struct FanOut;
            struct FanAvg;
        };
        using ScaleMode = ScaleModeTypeCate::FanAvg;
    };
    TypePolicyObj(PNormVarScale,    VarScaleFillerPolicy, Distribute, Norm);
    TypePolicyObj(PUniformVarScale, VarScaleFillerPolicy, Distribute, Uniform);
    TypePolicyObj(PVarScaleFanIn,   VarScaleFillerPolicy, ScaleMode,  FanIn);
    TypePolicyObj(PVarScaleFanOut,  VarScaleFillerPolicy, ScaleMode,  FanOut);
    TypePolicyObj(PVarScaleFanAvg,  VarScaleFillerPolicy, ScaleMode,  FanAvg);
}

#include <MetaNN/policies/policy_macro_end.h>
#endif //POLICIES_H
