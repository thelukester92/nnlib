#ifndef NN_CONTAINER_TPP
#define NN_CONTAINER_TPP

#include "../container.hpp"

namespace nnlib
{

template <typename T>
Container<T>::Container(const Container<T> &module) :
    Module<T>(module),
    m_components(module.m_components)
{
    for(Module<T> *&m : m_components)
    {
        /// \note intentionally not releasing; module still owns the original
        m = m->copy();
    }
}

template <typename T>
Container<T>::Container(const Serialized &node) :
    Module<T>(node),
    m_components(node.get<Storage<Module<T> *>>("components"))
{}

template <typename T>
Container<T> &Container<T>::operator=(const Container<T> &module)
{
    Module<T>::operator=(module);

    Storage<Module<T> *> components = module.m_components;
    for(Module<T> *&m : components)
    {
        /// \note intentionally not releasing; module still owns the original
        m = m->copy();
    }

    for(Module<T> *m : m_components)
        delete m;

    m_components = components;

    return *this;
}

template <typename T>
Container<T>::~Container()
{
    for(Module<T> *comp : m_components)
        delete comp;
}

template <typename T>
void Container<T>::training(bool training)
{
    for(Module<T> *comp : m_components)
        comp->training(training);
}

template <typename T>
void Container<T>::forget()
{
    Module<T>::forget();
    for(Module<T> *comp : m_components)
        comp->forget();
}

template <typename T>
void Container<T>::save(Serialized &node) const
{
    Module<T>::save(node);
    node.set("components", m_components);
}

template <typename T>
Module<T> *Container<T>::component(size_t index)
{
    return m_components[index];
}

template <typename T>
size_t Container<T>::components() const
{
    return m_components.size();
}

template <typename T>
Container<T> &Container<T>::add(Module<T> *component)
{
    m_components.push(component);
    return *this;
}

template <typename T>
Module<T> *Container<T>::remove(size_t index)
{
    Module<T> *comp = m_components[index];
    m_components.erase(index);
    return comp;
}

template <typename T>
Container<T> &Container<T>::clear()
{
    for(Module<T> *comp : m_components)
        delete comp;
    m_components.clear();
    return *this;
}

template <typename T>
Storage<Tensor<T> *> Container<T>::paramsList()
{
    Storage<Tensor<T> *> params;
    for(Module<T> *comp : m_components)
        params.append(comp->paramsList());
    return params;
}

template <typename T>
Storage<Tensor<T> *> Container<T>::gradList()
{
    Storage<Tensor<T> *> blams;
    for(Module<T> *comp : m_components)
        blams.append(comp->gradList());
    return blams;
}

template <typename T>
Storage<Tensor<T> *> Container<T>::stateList()
{
    Storage<Tensor<T> *> states;
    for(Module<T> *comp : m_components)
        states.append(comp->stateList());
    return states;
}

}

#endif
