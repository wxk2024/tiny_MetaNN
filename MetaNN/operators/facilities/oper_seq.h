//
// Created by wxk on 2024/10/28.
//

#ifndef OPER_SEQ_H
#define OPER_SEQ_H
namespace MetaNN
{
    // 求值过程的容器
    template <typename...TCases>
    struct OperSeqContainer;

    // 运算模板的求值逻辑：具体的计算方式
    template <typename TOpTag>
    struct OperSeq_;
}
#endif //OPER_SEQ_H
