#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "container.h"

namespace nnlib
{

template <typename T>
class Sequential : public Container<T>
{
using Container<T>::m_components;
public:
	virtual Matrix<T> &forward(const Matrix<T> &inputs) override
	{
		const Matrix<T> *inps = &inputs;
		for(Module<T> *layer : m_components)
			inps = &layer->forward(*inps);
		return m_components[0]->output();
	}
	
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		const Matrix<T> *blam = &blame;
		for(size_t i = m_components.size() - 1; i > 0; --i)
			blam = &m_components[i]->backward(m_components[i - 1]->output(), *blam);
		return m_components[0]->backward(inputs, *blam);
	}
	
	virtual Matrix<T> &output() override
	{
		return m_components[m_components.size() - 1]->output();
	}
	
	virtual Matrix<T> &inputBlame() override
	{
		return m_components[0]->inputBlame();
	}
};

}

#endif
