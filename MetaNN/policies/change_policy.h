//
// Created by wxk on 2024/11/1.
//

#ifndef CHANGE_POLICY_H
#define CHANGE_POLICY_H
#include <MetaNN/policies/policy_container.h>
namespace MetaNN{
    template <typename TNewPolicy, typename TOriContainer>
    struct ChangePolicy_;

    template <typename TNewPolicy, typename... TPolicies>
    struct ChangePolicy_<TNewPolicy, PolicyContainer<TPolicies...>>
    {
    private:
        using newMajor = typename TNewPolicy::MajorClass;
        using newMinor = typename TNewPolicy::MinorClass;

        template <typename TPC, typename... TP> struct DropAppend_;

        template <typename... TFilteredPolicies>
        struct DropAppend_<PolicyContainer<TFilteredPolicies...>>
        {
            using type = PolicyContainer<TFilteredPolicies..., TNewPolicy>; // 找到最后已经没有冲突的了，直接添加进来
        };

        template <typename... TFilteredPolicies, typename TCurPolicy, typename... TP>
        struct DropAppend_<PolicyContainer<TFilteredPolicies...>,
                           TCurPolicy, TP...>
        {
            template <bool isArray, typename TDummy = void>
            struct ArrayBasedSwitch_
            {
                template <typename TMajor, typename TMinor, typename TD = void>
                struct _impl
                {
                    using type = PolicyContainer<TFilteredPolicies..., TCurPolicy>;
                };

                template <typename TD>
                struct _impl<newMajor, newMinor, TD>
                {
                    using type = PolicyContainer<TFilteredPolicies...>; //由于不能冲突，所以就丢掉了
                };
                using type = typename _impl<typename TCurPolicy::MajorClass,
                                            typename TCurPolicy::MinorClass>::type;
            };

            template <typename TDummy>
            struct ArrayBasedSwitch_<true, TDummy>
            {
                using type = PolicyContainer<TFilteredPolicies..., TCurPolicy>; // IsSubPolicyContainer<TCurPolicy> 为真，则直接扔进去
            };

            using t1 = typename ArrayBasedSwitch_<IsSubPolicyContainer<TCurPolicy>>::type; //会丢掉和 new policy 冲突的
            using type = typename DropAppend_<t1, TP...>::type; // 丢掉后合法的是 t1 ，然后继续看 TP... 里面有没有不合法的
        };

    public:
        using type = typename DropAppend_<PolicyContainer<>, TPolicies...>::type;
    };

    // -----------TOriContainer中有TNewPolicy冲突的，就改成TNewPolicy；否则添加TNewPolicy--------
    template <typename TNewPolicy, typename TOriContainer>
    using ChangePolicy = typename ChangePolicy_<TNewPolicy, TOriContainer>::type;
}
#endif //CHANGE_POLICY_H
