//
// Created by wxk on 2024/11/1.
//

#ifndef LAYER_POLICIES_H
#define LAYER_POLICIES_H
#include <MetaNN/policies/policy_macro_begin.h>
#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/policies/inject_policy.h>
#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/interface_fun.h>
namespace MetaNN {
    // ------------------反向传播的策略------------------
    struct FeedbackPolicy {
        using MajorClass = FeedbackPolicy;
        struct IsUpdateValueCate;           // minor class:是否要对某个层的参数进行更新
        struct IsFeedbackOutputValueCate;   // minor class:该层是否需要计算输出梯度

        static constexpr bool IsUpdate = false;
        static constexpr bool IsFeedbackOutput = false;
    };
    ValuePolicyObj(PUpdate,   FeedbackPolicy, IsUpdate, true);
    ValuePolicyObj(PNoUpdate, FeedbackPolicy, IsUpdate, false);
    ValuePolicyObj(PFeedbackOutput,   FeedbackPolicy, IsFeedbackOutput, true);
    ValuePolicyObj(PFeedbackNoOutput, FeedbackPolicy, IsFeedbackOutput, false);

    // ------------------前向传播输入值的策略------------------
    struct InputPolicy
    {
        using MajorClass = InputPolicy;

        struct BatchModeValueCate;
        static constexpr bool BatchMode = false;
    };
    ValuePolicyObj(PBatchMode,  InputPolicy, BatchMode, true);
    ValuePolicyObj(PNoBatchMode,InputPolicy, BatchMode, false);

    // ------------------操作在那个平台上的策略，操作在什么类型数字的策略------------------
    // 可以扩展到支持其他类型的计算单元 double 或者其他的计算设备 GPU
    struct OperandPolicy
    {
        using MajorClass = OperandPolicy;

        struct DeviceTypeCate : public MetaNN::DeviceTags {}; //
        using Device = DeviceTypeCate::CPU;

        struct ElementTypeCate;
        using Element = float;
    };
    TypePolicyObj(PCPUDevice, OperandPolicy, Device, CPU);
    TypePolicyTemplate(PElementTypeIs, OperandPolicy, Element);

    //------------------单层的策略------------------
    struct SingleLayerPolicy
    {
        using MajorClass = SingleLayerPolicy;

        struct ActionTypeCate
        {
            struct Sigmoid;
            struct Tanh;
        };
        struct HasBiasValueCate;

        using Action = ActionTypeCate::Sigmoid;
        static constexpr bool HasBias = true;
    };
    TypePolicyObj(PSigmoidAction, SingleLayerPolicy, Action, Sigmoid);
    TypePolicyObj(PTanhAction, SingleLayerPolicy, Action, Tanh);
    ValuePolicyObj(PBiasSingleLayer,  SingleLayerPolicy, HasBias, true);
    ValuePolicyObj(PNoBiasSingleLayer, SingleLayerPolicy, HasBias, false);

    //------------------卷积层的策略------------------
    struct RecurrentLayerPolicy
    {
        using MajorClass = RecurrentLayerPolicy;

        struct StepTypeCate
        {
            struct GRU;
        };
        struct UseBpttValueCate;

        using Step = StepTypeCate::GRU;
        constexpr static bool UseBptt = true;
    };
    TypePolicyObj(PRecGRUStep, RecurrentLayerPolicy, Step, GRU);
    ValuePolicyObj(PEnableBptt,  RecurrentLayerPolicy, UseBptt, true);
    ValuePolicyObj(PDisableBptt,  RecurrentLayerPolicy, UseBptt, false);
}
#include <MetaNN/policies/policy_macro_end.h>
#endif //POLICIES_H
