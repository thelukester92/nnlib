#ifndef DROPCONNECT_TPP
#define DROPCONNECT_TPP

#include "../dropconnect.hpp"
#include "nnlib/math/math.hpp"

namespace nnlib
{

template <typename T>
DropConnect<T>::DropConnect(Module<T> *module, T dropProbability) :
    Module<T>(module->inputShape(), module->outputShape()),
    m_module(module),
    m_dropProbability(dropProbability),
    m_training(true)
{
    NNAssertGreaterThanOrEquals(dropProbability, 0, "Expected a probability!");
    NNAssertLessThan(dropProbability, 1, "Expected a probability!");
}

template <typename T>
DropConnect<T>::DropConnect(const DropConnect<T> &module) :
    Module<T>(module),
    m_module(module.m_module->copy()),
    m_dropProbability(module.m_dropProbability),
    m_training(module.m_training)
{}

template <typename T>
DropConnect<T>::DropConnect(const Serialized &node) :
    Module<T>(node),
    m_module(node.get<Module<T> *>("module")),
    m_dropProbability(node.get<T>("dropProbability")),
    m_training(node.get<bool>("training"))
{}

template <typename T>
DropConnect<T> &DropConnect<T>::operator=(DropConnect<T> module)
{
    Module<T>::operator=(module);
    swap(*this, module);
    return *this;
}

template <typename T>
DropConnect<T>::~DropConnect()
{
    delete m_module;
}

template <typename T>
void swap(DropConnect<T> &a, DropConnect<T> &b)
{
    using std::swap;
    swap(a.m_module, b.m_module);
    swap(a.m_dropProbability, b.m_dropProbability);
    swap(a.m_training, b.m_training);
}

/// Get the module this is decorating.
template <typename T>
Module<T> &DropConnect<T>::module()
{
    return *m_module;
}

/// Set the module this is decorating.
template <typename T>
DropConnect<T> &DropConnect<T>::module(Module<T> *module)
{
    delete m_module;
    m_module = module;
    return *this;
}

/// Get the probability that an output is not dropped.
template <typename T>
T DropConnect<T>::dropProbability() const
{
    return m_dropProbability;
}

/// Set the probability that an output is not dropped.
template <typename T>
DropConnect<T> &DropConnect<T>::dropProbability(T dropProbability)
{
    NNAssertGreaterThanOrEquals(dropProbability, 0, "Expected a probability!");
    NNAssertLessThan(dropProbability, 1, "Expected a probability!");
    m_dropProbability = dropProbability;
    return *this;
}

template <typename T>
bool DropConnect<T>::isTraining() const
{
    return m_training;
}

template <typename T>
void DropConnect<T>::training(bool training)
{
    m_training = training;
    m_module->training(training);
}

template <typename T>
void DropConnect<T>::forget()
{
    m_module->forget();
}

// MARK: Serialization

template <typename T>
void DropConnect<T>::save(Serialized &node) const
{
    Module<T>::save(node);
    node.set("module", m_module);
    node.set("dropProbability", m_dropProbability);
    node.set("training", m_training);
}

// MARK: Computation

template <typename T>
Tensor<T> &DropConnect<T>::forward(const Tensor<T> &input)
{
    if(m_training)
    {
        NNAssert(input.dims() == 1 || input.dims() == 2, "Expected a vector or a matrix!");
        m_backup.resize(m_module->params().size());
        m_backup.copy(m_module->params());

        if(input.dims() == 1)
        {
            m_mask.resize(m_module->params().size());
            math::pointwiseProduct(math::bernoulli(m_mask, 1 - m_dropProbability), m_module->params());
            m_output = m_module->forward(input);
            m_module->params().copy(m_backup);
        }
        else
        {
            m_output.resize(input.size(0), 1);
            m_mask.resize(input.size(0), m_module->params().size());
            math::bernoulli(m_mask, 1 - m_dropProbability);
            for(size_t i = 0; i < input.size(0); ++i)
            {
                math::pointwiseProduct(m_backup, m_mask.select(0, i), m_module->params());
                m_module->forward(input.select(0, i));
                NNAssertEquals(m_module->output().dims(), 1, "Expected a vector!");
                m_output.resizeDim(1, m_module->output().size());
                m_output.select(0, i).copy(m_module->output());
            }
            m_module->params().copy(m_backup);
        }
    }
    else
        m_output = m_module->forward(input).scale(1 - m_dropProbability);

    return m_output;
}

template <typename T>
Tensor<T> &DropConnect<T>::backward(const Tensor<T> &input, const Tensor<T> &outGrad)
{
    if(m_training)
    {
        NNAssert(input.dims() == 1 || input.dims() == 2, "Expected a vector or a matrix!");
        m_backup.resize(m_module->params().size());
        m_backup.copy(m_module->params());

        if(input.dims() == 1)
        {
            math::pointwiseProduct(m_mask, m_module->params());
            m_inGrad = m_module->backward(input, outGrad);
            m_module->params().copy(m_backup);
        }
        else
        {
            m_inGrad.resize(input.size(0), 1);
            for(size_t i = 0; i < input.size(0); ++i)
            {
                math::pointwiseProduct(m_backup, m_mask.select(0, i), m_module->params());
                m_module->backward(input.select(0, i), outGrad.select(0, i));
                NNAssertEquals(m_module->inGrad().dims(), 1, "Expected a vector!");
                m_inGrad.resizeDim(1, m_module->inGrad().size());
                m_inGrad.select(0, i).copy(m_module->inGrad());
            }
            m_module->params().copy(m_backup);
        }
    }
    else
        m_inGrad = m_module->backward(input, outGrad).scale(1 - m_dropProbability);

    return m_inGrad;
}

// MARK: Buffers

template <typename T>
Storage<Tensor<T> *> DropConnect<T>::paramsList()
{
    return m_module->paramsList();
}

template <typename T>
Storage<Tensor<T> *> DropConnect<T>::gradList()
{
    return m_module->gradList();
}

template <typename T>
Storage<Tensor<T> *> DropConnect<T>::stateList()
{
    return Module<T>::stateList().append(m_module->stateList()).append({ &m_mask, &m_backup });
}

}

#endif
