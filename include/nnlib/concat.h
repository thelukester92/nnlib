#ifndef CONCAT_H
#define CONCAT_H

#include "container.h"

namespace nnlib
{

template <typename T = double>
class Concat : public Container<T>
{
using Container<T>::m_components;
public:
	Concat() {}
	
	template <typename ... Ts>
	Concat(Ts*...more)
	{
		add(more...);
	}
	
	virtual void add(Module<T> *component) override
	{
		NNHardAssert(m_components.size() == 0 || component->inputCount() == m_components[0]->inputCount(), "Incompatible concat component!");
		Container<T>::add(component);
		
		// Flatten all output matrices into a single output matrix
		m_outputs.resize(component->batchSize(), m_outputs.cols() + component->outputCount());
		size_t offset = 0;
		for(auto *c : m_components)
		{
			m_outputs.block(c->output(), 0, offset, (size_t) -1, c->outputCount());
			offset += c->outputCount();
		}
		
		// Same-size input blame
		m_inputBlame.resize(component->batchSize(), component->inputCount());
	}
	
	template <typename ... Ts>
	void add(Module<T> *component, Ts*...more)
	{
		add(component);
		add(more...);
	}
	
	virtual void resize(size_t inps, size_t outs, size_t bats) override
	{
		NNHardAssert(m_components.size() > 0, "Cannot resize an empty concat module!");
		NNHardAssert(outs == m_outputs.cols(), "Cannot directly change output size of a concat module!");
		for(auto *c : m_components)
		{
			c->resize(inps);
			c->batch(bats);
		}
		m_inputBlame.resize(m_components[0]->inputBlame().rows(), m_components[0]->inputBlame().cols());
	}
	
	virtual void batch(size_t size) override
	{
		m_outputs.resize(size, m_outputs.cols());
		m_inputBlame.resize(size, m_inputBlame.cols());
		size_t offset = 0;
		for(auto *c : m_components)
		{
			c->batch(size);
			m_outputs.block(c->output(), 0, offset, (size_t) -1, c->outputCount());
			offset += c->outputCount();
		}
	}
	
	virtual Matrix<T> &forward(const Matrix<T> &inputs) override
	{
		for(auto *c : m_components)
			c->forward(inputs);
		return m_outputs;
	}
	
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		m_inputBlame.fill(0);
		
		Matrix<T> &ncBlame = *const_cast<Matrix<T> *>(&blame);
		
		size_t offset = 0;
		for(auto *c : m_components)
		{
			Matrix<T> blam = ncBlame.block(0, offset, (size_t) -1, c->outputCount());
			m_inputBlame.add(c->backward(inputs, blam));
			offset += c->outputCount();
		}
		
		return m_inputBlame;
	}
	
	virtual Matrix<T> &output() override
	{
		return m_outputs;
	}
	
	virtual Matrix<T> &inputBlame() override
	{
		return m_inputBlame;
	}
	
	virtual Vector<Tensor<T> *> parameters() override
	{
		Vector<Tensor<T> *> params;
		for(Module<T> *layer : m_components)
			for(Tensor<T> *t : layer->parameters())
				params.push_back(t);
		return params;
	}
	
	virtual Vector<Tensor<T> *> blame() override
	{
		Vector<Tensor<T> *> blam;
		for(Module<T> *layer : m_components)
			for(Tensor<T> *t : layer->blame())
				blam.push_back(t);
		return blam;
	}
	
private:
	Matrix<T> m_outputs, m_inputBlame;
};

}

#endif
