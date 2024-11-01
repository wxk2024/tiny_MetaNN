//
// Created by wxk on 2024/10/20.
//
// --------------- 四个参数的用来创建具体的 policy 对象 -----------------
#define TypePolicyObj(PolicyName, Ma, Mi, Val) \
struct PolicyName : virtual public Ma\
{ \
using MinorClass = Ma::Mi##TypeCate; \
using Mi = Ma::Mi##TypeCate::Val; \
}

#define ValuePolicyObj(PolicyName, Ma, Mi, Val) \
struct PolicyName : virtual public Ma \
{ \
using MinorClass = Ma::Mi##ValueCate; \
private: \
using type1 = decltype(Ma::Mi); \
using type2 = RemConstRef<type1>; \
public: \
static constexpr type2 Mi = static_cast<type2>(Val); \
}

// --------------- 三个参数的用(policy模板)来创建 policy 模板 -----------------
#define TypePolicyTemplate(PolicyName, Ma, Mi) \
template <typename T> \
struct PolicyName : virtual public Ma \
{ \
using MinorClass = Ma::Mi##TypeCate; \
using Mi = T; \
}

#define ValuePolicyTemplate(PolicyName, Ma, Mi) \
template <RemConstRef<decltype(Ma::Mi)> T> \
struct PolicyName : virtual public Ma \
{ \
using MinorClass = Ma::Mi##ValueCate; \
private: \
using type1 = decltype(Ma::Mi); \
using type2 = RemConstRef<type1>; \
public: \
static constexpr type2 Mi = T; \
}



