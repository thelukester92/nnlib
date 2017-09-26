#ifndef NN_LINEAR_H
#define NN_LINEAR_H

#include "module.h"

namespace nnlib
{

/// A standard feed-forward layer that returns a linear combination of inputs.
template <typename T = double>
class Linear : public Module<T>
{
public:
	Linear(size_t inps, size_t outs) :
		m_weights(inps, outs),
		m_weightsGrad(inps, outs),
		m_bias(outs),
		m_biasGrad(outs)
	{
		reset();
	}
	
	Linear(const Linear &module) :
		m_weights(module.m_weights.copy()),
		m_weightsGrad(m_weights.shape()),
		m_bias(module.m_bias.copy()),
		m_biasGrad(m_bias.shape())
	{}
	
	Linear(const Serialized &node) :
		m_weights(node.get<Tensor<T>>("weights")),
		m_weightsGrad(m_weights.shape()),
		m_bias(node.get<Tensor<T>>("bias")),
		m_biasGrad(m_bias.shape())
	{
		NNAssertEquals(m_weights.dims(), 2, "Expected matrix weights!");
		NNAssertEquals(m_bias.dims(), 1, "Expected vector bias!");
		NNAssertEquals(m_weights.size(1), m_bias.size(), "Incompatible weights and bias!");
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
		swap(a.m_bias, b.m_bias);
		swap(a.m_biasGrad, b.m_biasGrad);
	}
	
	Linear &reset()
	{
		if(m_weights.size() > 0)
		{
			T dev = 1.0 / sqrt(m_weights.size(1));
			m_weights.rand(-dev, dev);
			m_bias.rand(-dev, dev);
		}
		return *this;
	}
	
	Tensor<T> weights()
	{
		return m_weights;
	}
	
	Tensor<T> bias()
	{
		return m_bias;
	}
	
	// MARK: Serialization
	
	virtual void save(Serialized &node) override
	{
		node.set("weights", m_weights);
		node.set("bias", m_bias);
	}
	
	// MARK: Computation
	
	virtual void updateOutput(const Tensor<T> &input) override
	{
		if(input.dims() == 1)
		{
			m_output.resize(m_weights.size(1));
			m_output.copy(m_bias).assignMTV(m_weights, input, 1, 1);
		}
		else if(input.dims() == 2)
		{
			m_output.resize(input.size(0), m_weights.size(1));
			m_output.assignMM(input, m_weights);
			m_output.assignVV(m_ones.resize(input.size(0)).fill(1), m_bias, 1, 1);
		}
		else
		{
			throw Error("Expected vector or matrix input!");
		}
	}
	
	virtual void updateInGrad(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssertEquals(input.dims(), outGrad.dims(), "Incompatible input and outGrad!");
		if(input.dims() == 1)
		{
			m_inGrad.resize(m_weights.size(0));
			m_inGrad.assignMV(m_weights, outGrad);
		}
		else if(input.dims() == 2)
		{
			m_inGrad.resize(input.size(0), m_weights.size(0));
			m_inGrad.assignMMT(outGrad, m_weights);
		}
		else
		{
			throw Error("Expected vector or matrix input and outGrad!");
		}
	}
	
	virtual void updateParamsGrad(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssertEquals(input.dims(), outGrad.dims(), "Incompatible input and outGrad!");
		if(input.dims() == 1)
		{
			m_weightsGrad.assignVV(input, outGrad, 1, 1);
			m_biasGrad.addV(outGrad);
		}
		else if(input.dims() == 2)
		{
			m_weightsGrad.assignMTM(input, outGrad, 1, 1);
			m_biasGrad.assignMTV(outGrad, m_ones.resize(input.size(0)).fill(1), 1, 1);
		}
		else
		{
			throw Error("Expected vector or matrix input and outGrad!");
		}
	}
	
	// MARK: Buffers
	
	virtual Storage<Tensor<T> *> paramsList() override
	{
		return { &m_weights, &m_bias };
	}
	
	virtual Storage<Tensor<T> *> gradList() override
	{
		return { &m_weightsGrad, &m_biasGrad };
	}
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
	
private:
	Tensor<T> m_weights;
	Tensor<T> m_weightsGrad;
	Tensor<T> m_bias;
	Tensor<T> m_biasGrad;
	
	Tensor<T> m_ones;
};

}

NNRegisterType(Linear, Module);

#endif
