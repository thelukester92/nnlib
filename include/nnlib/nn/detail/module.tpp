#ifndef NN_MODULE_TPP
#define NN_MODULE_TPP

#include "../module.hpp"
#include "nnlib/serialization/factory.hpp"

namespace nnlib
{

template <typename T>
Module<T>::Module() :
    m_inGrad(1),
    m_output(1)
{}

template <typename T>
Module<T>::Module(const std::initializer_list<size_t> &ioShape) :
    Module(Storage<size_t>(ioShape))
{}

template <typename T>
Module<T>::Module(const Storage<size_t> &ioShape) :
    m_inGrad(ioShape, true),
    m_output(ioShape, true)
{}

template <typename T>
Module<T>::Module(const Storage<size_t> &inputShape, const Storage<size_t> &outputShape) :
    m_inGrad(inputShape, true),
    m_output(outputShape, true)
{}

template <typename T>
Module<T>::Module(const Serialized &node) :
    m_inGrad(node.get<Storage<size_t>>("inputShape"), true),
    m_output(node.get<Storage<size_t>>("outputShape"), true)
{}

template <typename T>
Module<T>::Module(const Module<T> &module) :
    m_inGrad(module.inputShape(), true),
    m_output(module.outputShape(), true)
{}

template <typename T>
Module<T>::~Module()
{}

template <typename T>
Module<T> &Module<T>::operator=(const Module<T> &module)
{
    m_inGrad.resize(module.inputShape());
    m_output.resize(module.outputShape());
    return *this;
}

template <typename T>
Module<T> *Module<T>::copy() const
{
    return Factory<Module<T>>::constructCopy(this);
}

template <typename T>
void Module<T>::training(bool training)
{}

template <typename T>
void Module<T>::forget()
{
    state().fill(0);
}

template <typename T>
void Module<T>::save(Serialized &node) const
{
    node.set("inputShape", m_inGrad.shape());
    node.set("outputShape", m_output.shape());
}

template <typename T>
Storage<Tensor<T> *> Module<T>::paramsList()
{
    return {};
}

template <typename T>
Storage<Tensor<T> *> Module<T>::gradList()
{
    return {};
}

template <typename T>
Storage<Tensor<T> *> Module<T>::stateList()
{
    return { &m_output };
}

template <typename T>
Tensor<T> &Module<T>::params()
{
    auto list = paramsList();
    if(!m_params.sharedWith(list))
        m_params = Tensor<T>::vectorize(list);
    return m_params;
}

template <typename T>
Tensor<T> &Module<T>::grad()
{
    auto list = gradList();
    if(!m_grad.sharedWith(list))
        m_grad = Tensor<T>::vectorize(list);
    return m_grad;
}

template <typename T>
Tensor<T> &Module<T>::state()
{
    auto list = stateList();
    if(!m_state.sharedWith(list))
        m_state = Tensor<T>::vectorize(list);
    return m_state;
}

template <typename T>
Tensor<T> &Module<T>::output()
{
    return m_output;
}

template <typename T>
const Tensor<T> &Module<T>::output() const
{
    return const_cast<Module<T> *>(this)->output();
}

template <typename T>
Tensor<T> &Module<T>::inGrad()
{
    return m_inGrad;
}

template <typename T>
const Tensor<T> &Module<T>::inGrad() const
{
    return const_cast<Module<T> *>(this)->inGrad();
}

template <typename T>
const Storage<size_t> &Module<T>::inputShape() const
{
    return m_inGrad.shape();
}

template <typename T>
const Storage<size_t> &Module<T>::outputShape() const
{
    return m_output.shape();
}

}

#endif
