#ifndef NN_SEQUENCER_TPP
#define NN_SEQUENCER_TPP

#include "../sequencer.hpp"

namespace nnlib
{

template <typename T>
Sequencer<T>::Sequencer(Module<T> *module, bool reverse) :
    m_module(module),
    m_reverse(reverse)
{}

template <typename T>
Sequencer<T>::Sequencer(const Sequencer<T> &module) :
    m_module(module.m_module->copy()),
    m_reverse(module.m_reverse)
{}

template <typename T>
Sequencer<T>::Sequencer(const Serialized &node) :
    m_module(node.get<Module<T> *>("module")),
    m_reverse(node.get<bool>("reverse"))
{}

template <typename T>
Sequencer<T>::~Sequencer()
{
    delete m_module;
}

template <typename T>
Sequencer<T> &Sequencer<T>::operator=(Sequencer<T> module)
{
    swap(*this, module);
    return *this;
}

template <typename T>
void swap(Sequencer<T> &a, Sequencer<T> &b)
{
    using std::swap;
    swap(a.m_module, b.m_module);
    swap(a.m_reverse, b.m_reverse);
}

template <typename T>
Module<T> &Sequencer<T>::module()
{
    return *m_module;
}

template <typename T>
Sequencer<T> &Sequencer<T>::module(Module<T> *module)
{
    delete m_module;
    m_module = module;
    return *this;
}

template <typename T>
bool Sequencer<T>::isReversed()
{
    return m_reverse;
}

template <typename T>
Sequencer<T> &Sequencer<T>::reverse(bool reverse)
{
    m_reverse = reverse;
    return *this;
}

template <typename T>
void Sequencer<T>::startForward(const Tensor<T> &first, size_t sequenceLength)
{
    m_module->forward(first);

    m_output.resize(Storage<size_t>({ sequenceLength }).append(m_module->output().shape()));
    if(m_reverse)
        m_output.select(0, sequenceLength - 1).copy(m_module->output());
    else
        m_output.select(0, 0).copy(m_module->output());

    m_states.resize(Storage<size_t>({ sequenceLength }).append(m_module->state().shape()));
    if(m_reverse)
        m_states.select(0, sequenceLength - 1).copy(m_module->state());
    else
        m_states.select(0, 0).copy(m_module->state());
}

template <typename T>
void Sequencer<T>::stepForward(const Tensor<T> &singleInput, size_t i)
{
    m_output.select(0, i).copy(m_module->forward(singleInput));
    m_states.select(0, i).copy(m_module->state());
}

template <typename T>
void Sequencer<T>::stepBackward(const Tensor<T> &singleInput, const Tensor<T> &singleOutGrad, size_t i)
{
    m_module->state().copy(m_states.select(0, i));
    m_inGrad.select(0, i).copy(m_module->backward(singleInput, singleOutGrad));
}

template <typename T>
void Sequencer<T>::training(bool training)
{
    m_module->training(training);
}

template <typename T>
void Sequencer<T>::forget()
{
    Module<T>::forget();
    m_module->forget();
}

template <typename T>
void Sequencer<T>::save(Serialized &node) const
{
    node.set("module", m_module);
    node.set("reverse", m_reverse);
}

template <typename T>
Tensor<T> &Sequencer<T>::forward(const Tensor<T> &input)
{
    if(m_reverse)
    {
        size_t len = input.size(0);
        startForward(input.select(0, len - 1), len);
        for(size_t i = len - 1; i > 0; --i)
            stepForward(input.select(0, i - 1), i - 1);
    }
    else
    {
        startForward(input.select(0, 0), input.size(0));
        for(size_t i = 1, end = input.size(0); i < end; ++i)
            stepForward(input.select(0, i), i);
    }

    return m_output;
}

template <typename T>
Tensor<T> &Sequencer<T>::backward(const Tensor<T> &input, const Tensor<T> &outGrad)
{
    NNAssertEquals(input.size(0), outGrad.size(0), "Incompatible input and outGrad!");
    NNAssertEquals(input.size(0), m_output.size(0), "Sequencer::forward must be called first!");
    m_inGrad.resize(input.shape());

    if(m_reverse)
    {
        for(size_t i = 0, len = input.size(0); i < len; ++i)
            stepBackward(input.select(0, i), outGrad.select(0, i), i);
    }
    else
    {
        for(size_t i = input.size(0); i > 0; --i)
            stepBackward(input.select(0, i - 1), outGrad.select(0, i - 1), i - 1);
    }

    return m_inGrad;
}

template <typename T>
Storage<Tensor<T> *> Sequencer<T>::paramsList()
{
    return m_module->paramsList();
}

template <typename T>
Storage<Tensor<T> *> Sequencer<T>::gradList()
{
    return m_module->gradList();
}

template <typename T>
Storage<Tensor<T> *> Sequencer<T>::stateList()
{
    return Module<T>::stateList().append(m_module->stateList()).push_back(&m_states);
}

}

#endif
