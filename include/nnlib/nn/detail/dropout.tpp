#ifndef NN_DROPOUT_TPP
#define NN_DROPOUT_TPP

#include "../dropout.hpp"
#include "nnlib/math/math.hpp"

namespace nnlib
{

template <typename T>
Dropout<T>::Dropout(T dropProbability) :
    m_dropProbability(dropProbability),
    m_training(true)
{
    NNAssertGreaterThanOrEquals(dropProbability, 0, "Expected a probability!");
    NNAssertLessThan(dropProbability, 1, "Expected a probability!");
}

template <typename T>
Dropout<T>::Dropout(const Dropout<T> &module) :
    m_dropProbability(module.m_dropProbability),
    m_training(module.m_training)
{}

template <typename T>
Dropout<T>::Dropout(const Serialized &node) :
    m_dropProbability(node.get<T>("dropProbability")),
    m_training(node.get<bool>("training"))
{}

template <typename T>
Dropout<T> &Dropout<T>::operator=(const Dropout<T> &module)
{
    m_dropProbability	= module.m_dropProbability;
    m_training			= module.m_training;
    return *this;
}

template <typename T>
T Dropout<T>::dropProbability() const
{
    return m_dropProbability;
}

template <typename T>
Dropout<T> &Dropout<T>::dropProbability(T dropProbability)
{
    NNAssertGreaterThanOrEquals(dropProbability, 0, "Expected a probability!");
    NNAssertLessThan(dropProbability, 1, "Expected a probability!");
    m_dropProbability = dropProbability;
    return *this;
}

template <typename T>
bool Dropout<T>::isTraining() const
{
    return m_training;
}

template <typename T>
void Dropout<T>::training(bool training)
{
    m_training = training;
}

template <typename T>
void Dropout<T>::save(Serialized &node) const
{
    node.set("dropProbability", m_dropProbability);
    node.set("training", m_training);
}

template <typename T>
Tensor<T> &Dropout<T>::forward(const Tensor<T> &input)
{
    m_mask.resize(input.shape());
    m_output.resize(input.shape());

    if(m_training)
        return math::pointwiseProduct(input, math::bernoulli(m_mask, 1 - m_dropProbability), m_output);
    else
        return m_output.copy(input).scale(1 - m_dropProbability);
}

template <typename T>
Tensor<T> &Dropout<T>::backward(const Tensor<T> &input, const Tensor<T> &outGrad)
{
    NNAssertEquals(input.shape(), m_mask.shape(), "Dropout<T>::forward must be called first!");
    NNAssertEquals(input.shape(), outGrad.shape(), "Incompatible input and outGrad!");
    m_inGrad.resize(input.shape());

    if(m_training)
        return math::pointwiseProduct(outGrad, m_mask, m_inGrad);
    else
        return m_inGrad.copy(outGrad).scale(1 - m_dropProbability);
}

template <typename T>
Storage<Tensor<T> *> Dropout<T>::stateList()
{
    return Module<T>::stateList().push_back(&m_mask);
}

}

#endif
