//
// Created by wxk on 2024/10/19.
//

#ifndef POLICY_SELECTOR_H
#define POLICY_SELECTOR_H
#include "policy_container.h"

namespace MetaNN {
	namespace NSPolicySelect {
		// -------------实现不冲突的 policy 对象们的继承类的创建(多重继承)-------------
		template<typename TPolicyCont>
		struct PolicySelRes;

		template<typename TPolicy>
		struct PolicySelRes<PolicyContainer<TPolicy> > : public TPolicy {
		};

		template<typename TCurPolicy, typename... TOtherPolicies>
		struct PolicySelRes<PolicyContainer<TCurPolicy, TOtherPolicies...> >
				: public TCurPolicy, public PolicySelRes<PolicyContainer<TOtherPolicies...> > {
		};

		// 按照 major class 来挑选 policy,放到 MCO 容器中
		template<typename MCO, typename TMajorClass, typename... TP>
		struct MajorFilter_ {
			using type = MCO;
		};

		template<typename... TFilteredPolicies, typename TMajorClass,
			typename TCurPolicy, typename... TP>
		struct MajorFilter_<PolicyContainer<TFilteredPolicies...>, TMajorClass,
					TCurPolicy, TP...> {
			// 这个地方的 TDummy 参数，仅仅是为了应对 C++ 当中模板类中，不能有模板偏特化的限制
			template<typename CurMajor, typename TDummy = void>
			struct _impl // CurMajor 不满足 TMarjorClass 的要求，拿掉它，不让他进入PolicyContainer
			{
				using type = typename MajorFilter_<PolicyContainer<TFilteredPolicies...>, TMajorClass, TP...>::type;
			};

			template<typename TDummy> // CurMajor 满足 TMarjorClass 的要求，
			struct _impl<TMajorClass, TDummy> {
				using type = typename MajorFilter_<PolicyContainer<TFilteredPolicies..., TCurPolicy>,
					TMajorClass, TP...>::type;
			};

			using type = typename _impl<typename TCurPolicy::MajorClass>::type;
		};

		/// ================= Minor Check ===============================
		template<typename TMinorClass, typename... TP>
		struct MinorDedup_ {
			static constexpr bool value = true;
		};
		template <typename TMinorClass, typename TCurPolicy, typename... TP>
		struct MinorDedup_<TMinorClass, TCurPolicy, TP...>
		{
			using TCurMirror = typename TCurPolicy::MinorClass;
			constexpr static bool cur_check = !(std::is_same<TMinorClass, TCurMirror>::value);
			constexpr static bool value = AndValue<cur_check,
												   MinorDedup_<TMinorClass, TP...>>;
		};
		template <typename TPolicyCont>
		struct MinorCheck_
		{
			static constexpr bool value = true;
		};
		template <typename TCurPolicy, typename... TP>
		struct MinorCheck_<PolicyContainer<TCurPolicy, TP...>>
		{
			static constexpr bool cur_check = MinorDedup_<typename TCurPolicy::MinorClass, TP...>::value;

			static constexpr bool value
				= AndValue<cur_check, MinorCheck_<PolicyContainer<TP...>>>;
		};

		template <typename TMajorClass, typename TPolicyContainer>
		struct Selector_;

		template <typename TMajorClass, typename... TPolicies>
		struct Selector_<TMajorClass, PolicyContainer<TPolicies...>>
		{
			using TMF = typename MajorFilter_<PolicyContainer<>, TMajorClass, TPolicies...>::type; // 将符合的都扔到 PolicyContainer 当中
			static_assert(MinorCheck_<TMF>::value, "Minor class set conflict!");   //只有这样才可以有良好的提示信息

			using type = std::conditional_t<IsArrayEmpty<TMF>, TMajorClass, PolicySelRes<TMF>>; //要不就是默认的,要不就是继承好的类型
		};
	}
	template <typename TMajorClass, typename TPolicyContainer>
	using PolicySelect = typename NSPolicySelect::Selector_<TMajorClass, TPolicyContainer>::type;
}

#endif //POLICY_SELECTOR_H
