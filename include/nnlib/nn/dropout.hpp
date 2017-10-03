#ifndef DROPOUT_HPP
#define DROPOUT_HPP

#include "module.hpp"

namespace nnlib
{

template <typename T = double>
class Dropout : public Module<T>
{
public:
	Dropout(T dropProbability = 0.1) :
		m_dropProbability(dropProbability),
		m_training(true)
	{
		NNAssertGreaterThanOrEquals(dropProbability, 0, "Expected a probability!");
		NNAssertLessThan(dropProbability, 1, "Expected a probability!");
	}
	
	Dropout(const Dropout &module) :
		m_dropProbability(module.m_dropProbability),
		m_training(module.m_training)
	{}
	
	Dropout(const Serialized &node) :
		m_dropProbability(node.get<T>("dropProbability")),
		m_training(node.get<bool>("training"))
	{}
	
	Dropout &operator=(const Dropout &module)
	{
		m_dropProbability	= module.m_dropProbability;
		m_training			= module.m_training;
		return *this;
	}
	
	/// Get the probability that an output is not dropped.
	T dropProbability() const
	{
		return m_dropProbability;
	}
	
	/// Set the probability that an output is not dropped.
	Dropout &dropProbability(T dropProbability)
	{
		NNAssertGreaterThanOrEquals(dropProbability, 0, "Expected a probability!");
		NNAssertLessThan(dropProbability, 1, "Expected a probability!");
		m_dropProbability = dropProbability;
		return *this;
	}
	
	bool isTraining() const
	{
		return m_training;
	}
	
	virtual void training(bool training = true) override
	{
		m_training = training;
	}
	
	// MARK: Serialization
	
	virtual void save(Serialized &node) const override
	{
		node.set("dropProbability", m_dropProbability);
		node.set("training", m_training);
	}
	
	// MARK: Computation
	
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		m_mask.resize(input.shape());
		m_output.resize(input.shape());
		
		if(m_training)
			return m_output.copy(input).pointwiseProduct(m_mask.bernoulli(1 - m_dropProbability));
		else
			return m_output.copy(input).scale(1 - m_dropProbability);
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssertEquals(input.shape(), m_mask.shape(), "Dropout::forward must be called first!");
		NNAssertEquals(input.shape(), outGrad.shape(), "Incompatible input and outGrad!");
		m_inGrad.resize(input.shape());
		
		if(m_training)
			return m_inGrad.copy(outGrad).pointwiseProduct(m_mask);
		else
			return m_inGrad.copy(outGrad).scale(1 - m_dropProbability);
	}
	
	// MARK: Buffers
	
	virtual Storage<Tensor<T> *> stateList() override
	{
		return Module<T>::stateList().push_back(&m_mask);
	}
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
	
private:
	Tensor<T> m_mask;
	T m_dropProbability;
	bool m_training;
};

}

NNRegisterType(Dropout, Module);

#endif
