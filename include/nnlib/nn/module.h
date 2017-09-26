#ifndef NN_MODULE_H
#define NN_MODULE_H

#include "../core/tensor.h"
#include "../serialization/factory.h"
#include "../serialization/serialized.h"

namespace nnlib
{

/// The abtract base class for all neural network modules.
template <typename T = double>
class Module
{
public:
	// MARK: Serialization
	
	virtual void save(Serialized &) const = 0;
	
	// MARK: Computation
	
	virtual void updateOutput(const Tensor<T> &input) = 0;
	virtual void updateGrad(const Tensor<T> &input, const Tensor<T> &outGrad) = 0;
	
	Tensor<T> &forward(const Tensor<T> &input)
	{
		updateOutput(input);
		return m_output;
	}
	
	Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad)
	{
		updateGrad(input, outGrad);
		return m_inGrad;
	}
	
	// MARK: Buffers
	
	virtual Storage<Tensor<T> *> paramsList()
	{
		return {};
	}
	
	virtual Storage<Tensor<T> *> gradList()
	{
		return {};
	}
	
	virtual Storage<Tensor<T> *> stateList()
	{
		return { &m_output };
	}
	
	Tensor<T> &params()
	{
		auto list = paramsList();
		bool recalculate = false;
		
		for(size_t i = 0; i < list.size() && !recalculate; ++i)
			recalculate = !list[i]->sharedWith(m_params);
		
		if(recalculate)
			m_params = Tensor<T>::flatten(list);
		
		return m_params;
	}
	
	Tensor<T> &grad()
	{
		auto list = gradList();
		bool recalculate = false;
		
		for(size_t i = 0; i < list.size() && !recalculate; ++i)
			recalculate = !list[i]->sharedWith(m_grad);
		
		if(recalculate)
			m_grad = Tensor<T>::flatten(list);
		
		return m_grad;
	}
	
	Tensor<T> &state()
	{
		auto list = stateList();
		bool recalculate = false;
		
		for(size_t i = 0; i < list.size() && !recalculate; ++i)
			recalculate = !list[i]->sharedWith(m_state);
		
		if(recalculate)
			m_state = Tensor<T>::flatten(list);
		
		return m_state;
	}
	
	// MARK: Getters
	
	Tensor<T> &output()
	{
		return m_output;
	}
	
	const Tensor<T> &output() const
	{
		return m_output;
	}
	
	Tensor<T> &inGrad()
	{
		return m_inGrad;
	}
	
	const Tensor<T> &inGrad() const
	{
		return m_inGrad;
	}
	
protected:
	Tensor<T> m_output;
	Tensor<T> m_inGrad;
	
private:
	Tensor<T> m_params;
	Tensor<T> m_grad;
	Tensor<T> m_state;
};

}

#endif
