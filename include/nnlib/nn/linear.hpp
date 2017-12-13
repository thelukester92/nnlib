#ifndef NN_LINEAR_HPP
#define NN_LINEAR_HPP

#include "module.hpp"

namespace nnlib
{

/// A standard feed-forward layer that returns a linear combination of inputs.
template <typename T = double>
class Linear : public Module<T>
{
public:
	Linear(size_t inps, size_t outs, bool bias = true) :
		m_weights(inps, outs),
		m_weightsGrad(inps, outs),
		m_useBias(bias),
		m_bias(bias ? outs : 0),
		m_biasGrad(bias ? outs : 0)
	{
		reset();
	}
	
	Linear(const Linear &module) :
		m_weights(module.m_weights.copy()),
		m_weightsGrad(m_weights.shape(), true),
		m_useBias(module.m_useBias),
		m_bias(module.m_bias.copy()),
		m_biasGrad(m_bias.shape(), true)
	{}
	
	Linear(const Serialized &node) :
		m_weights(node.get<Tensor<T>>("weights")),
		m_weightsGrad(m_weights.shape(), true),
		m_useBias(node.get<bool>("useBias")),
		m_bias(node.get<Tensor<T>>("bias")),
		m_biasGrad(m_bias.shape(), true)
	{
		NNAssertEquals(m_weights.dims(), 2, "Expected matrix weights!");
		NNAssert(!m_useBias || m_bias.dims() == 1, "Expected vector bias!");
		NNAssert(!m_useBias || m_weights.size(1) == m_bias.size(), "Incompatible weights and bias!");
	}
	
	Linear &operator=(Linear module)
	{
		swap(*this, module);
		return *this;
	}
	
	friend void swap(Linear &a, Linear &b)
	{
		using std::swap;
		swap(a.m_weights, b.m_weights);
		swap(a.m_weightsGrad, b.m_weightsGrad);
		swap(a.m_useBias, b.m_useBias);
		swap(a.m_bias, b.m_bias);
		swap(a.m_biasGrad, b.m_biasGrad);
	}
	
	bool biased() const
	{
		return m_useBias;
	}
	
	Linear &reset()
	{
		T dev = 1.0 / sqrt(m_weights.size(1));
		m_weights.rand(-dev, dev);
		
		if(m_useBias)
			m_bias.rand(-dev, dev);
		
		return *this;
	}
	
	Tensor<T> weights()
	{
		return m_weights;
	}
	
	Tensor<T> bias()
	{
		NNHardAssert(m_useBias, "This is an unbiased module!");
		return m_bias;
	}
	
	// MARK: Serialization
	
	virtual void save(Serialized &node) const override
	{
		node.set("weights", m_weights);
		node.set("useBias", m_useBias);
		node.set("bias", m_bias);
	}
	
	// MARK: Computation
	
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		if(input.dims() == 1)
		{
			m_output.resize(m_weights.size(1));
			if(m_useBias)
				m_output.copy(m_bias).assignMTV(m_weights, input, 1, 1);
			else
				m_output.assignMTV(m_weights, input);
		}
		else if(input.dims() == 2)
		{
			m_output.resize(input.size(0), m_weights.size(1));
			m_output.assignMM(input, m_weights);
			if(m_useBias)
				m_output.assignVV(m_ones.resize(input.size(0)).fill(1), m_bias, 1, 1);
		}
		else
		{
			throw Error("Expected vector or matrix input!");
		}
		
		return m_output;
	}
	
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssertEquals(input.dims(), outGrad.dims(), "Incompatible input and outGrad!");
		if(input.dims() == 1)
		{
			m_weightsGrad.assignVV(input, outGrad, 1, 1);
			if(m_useBias)
				m_biasGrad.addV(outGrad);
			
			m_inGrad.resize(m_weights.size(0));
			m_inGrad.assignMV(m_weights, outGrad);
		}
		else if(input.dims() == 2)
		{
			m_weightsGrad.assignMTM(input, outGrad, 1, 1);
			if(m_useBias)
				m_biasGrad.assignMTV(outGrad, m_ones.resize(input.size(0)).fill(1), 1, 1);
			
			m_inGrad.resize(input.size(0), m_weights.size(0));
			m_inGrad.assignMMT(outGrad, m_weights);
		}
		else
		{
			throw Error("Expected vector or matrix input and outGrad!");
		}
		
		return m_inGrad;
	}
	
	// MARK: Buffers
	
	virtual Storage<Tensor<T> *> paramsList() override
	{
		if(m_useBias)
			return { &m_weights, &m_bias };
		else
			return { &m_weights };
	}
	
	virtual Storage<Tensor<T> *> gradList() override
	{
		if(m_useBias)
			return { &m_weightsGrad, &m_biasGrad };
		else
			return { &m_weightsGrad };
	}
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
	
	Tensor<T> m_weights;
	Tensor<T> m_weightsGrad;
	
	bool m_useBias;
	Tensor<T> m_bias;
	Tensor<T> m_biasGrad;
	
	Tensor<T> m_ones;
};

}

NNRegisterType(Linear, Module);

#endif
