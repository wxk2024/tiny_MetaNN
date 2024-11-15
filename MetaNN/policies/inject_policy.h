//
// Created by wxk on 2024/11/6.
//

#ifndef INJECT_POLICY_H
#define INJECT_POLICY_H
#include <MetaNN/policies/policy_container.h>
namespace MetaNN
{
    //template <typename...TParams>
    //template<template <typename TPolicyCont, typename...> class T, typename...TPolicies>
    //using InjectPolicy = T<PolicyContainer<TPolicies...>, TParams...>;

    template<template <typename TPolicyCont> class T, typename...TPolicies>
    using InjectPolicy = T<PolicyContainer<TPolicies...>>;
}
#endif //INJECT_POLICY_H
