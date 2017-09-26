#ifndef NN_LINEAR_H
#define NN_LINEAR_H

#include "module.h"

namespace nnlib
{

/// \brief A standard feed-forward layer that returns a linear combination of inputs.
///
/// Input and output shapes are always in terms of vectors, but if a matrix input is given
/// it will batch accelerate and return a matrix output.
template <typename T = double>
class Linear : public Module<T>
{
public:
	/// Standard inps -> outs layer.
	Linear(size_t inps, size_t outs) :
		Module<T>({ inps }, { outs }),
		m_weights(inps, outs),
		m_weightsGrad(inps, outs),
		m_bias(outs),
		m_biasGrad(outs)
	{
		reset();
	}
	
	/// any -> outs layer; adding to a sequential will set input size.
	Linear(size_t outs = 0) :
		Module<T>({ 0 }, { outs }),
		m_weights(0, outs),
		m_weightsGrad(0, outs),
		m_bias(outs),
		m_biasGrad(outs)
	{}
	
	Linear(const Linear &module) :
		Module<T>(module.inputShape(), module.outputShape()),
		m_weights(module.m_weights.copy()),
		m_weightsGrad(module.m_weightsGrad.copy()),
		m_bias(module.m_bias.copy()),
		m_biasGrad(module.m_biasGrad.copy())
	{}
	
	Linear &operator=(const Linear &module)
	{
		resizeOutputs(module.m_outputShape);
		resizeInputs(module.m_inputShape);
		m_weights		= module.m_weights.copy();
		m_weightsGrad	= module.m_weightsGrad.copy();
		m_bias			= module.m_bias.copy();
		m_biasGrad		= module.m_biasGrad.copy();
		m_addBuffer		= module.m_addBuffer.copy();
		return *this;
	}
	
	/// Set weights to uniformly distributed random values.
	Linear &reset()
	{
		if(m_weights.size() > 0)
		{
			T range = 1.0 / sqrt(m_weights.size(1));
			m_weights.rand(-range, range);
			m_bias.rand(-range, range);
		}
		return *this;
	}
	
	/// Get the weights of this module.
	const Tensor<T> &weights() const
	{
		return m_weights;
	}
	
	/// Get the bias of this module.
	const Tensor<T> &bias() const
	{
		return m_bias;
	}
	
	// MARK: Serialization
	
	/// Save to a serialized node.
	virtual void save(Serialized &node) const override
	{
		node.set("inputs", m_inputShape);
		node.set("outputs", m_outputShape);
		node.set("weights", m_weights);
		node.set("bias", m_bias);
	}
	
	/// Load from a serialized node.
	virtual void load(const Serialized &node) override
	{
		this->resize(node.get<Storage<size_t>>("inputs"), node.get<Storage<size_t>>("outputs"));
		node.get("weights", m_weights);
		node.get("bias", m_bias);
	}
	
	// MARK: Computation
	
	/// Forward propagate input, returning output.
	virtual const Tensor<T> &forward(const Tensor<T> &input) override
	{
		if(input.dims() == 2)
		{
			NNAssertEquals(input.select(0, 0).shape(), m_inputShape, "Incompatible input!");
			m_output.resize(input.size(0), m_outputShape[0]);
			
			// output (bats x outs) = input (bats x inps) x weights (inps x outs)
			m_output.assignMM(input, m_weights);
			
			// output (bats x outs) += addBuffer (bats x 1) x bias (1 x outs)
			m_output.assignVV(m_addBuffer.resize(input.size(0)).fill(1), m_bias, 1, 1);
		}
		else if(input.dims() == 1)
		{
			NNAssertEquals(input.shape(), m_inputShape, "Incompatible input!");
			m_output.resize(m_outputShape[0]);
			
			// output (outs) = weights^T (outs x inps) x input (inps)
			m_output.assignMTV(m_weights, input);
			
			// output (1 x outs) += bias (1 x outs)
			m_output.addV(m_bias);
		}
		else
		{
			throw Error("Incompatible input!");
		}
		
		return m_output;
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual const Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssertEquals(input.dims(), outGrad.dims(), "Incompatible input and outGrad!");
		
		if(input.dims() == 2)
		{
			NNAssertEquals(outGrad.select(0, 0).shape(), m_outputShape, "Incompatible outGrad!");
			NNAssertEquals(input.select(0, 0).shape(), m_inputShape, "Incompatible input!");
			m_inGrad.resize(input.size(0), m_inputShape[0]);
			
			// biasGrad (outs x 1) += outGrad^T (outs x bats) x addBuffer (bats x 1)
			m_biasGrad.assignMTV(outGrad, m_addBuffer.resize(input.size(0)), 1, 1);
			
			// weightsGrad (inps x outs) += input^T (bats x inps) x outGrad (bats x outs)
			m_weightsGrad.assignMTM(input, outGrad, 1, 1);
			
			// inGrad (bats x inps) = outGrad (bats x outs) x weights^T (outs x inps)
			m_inGrad.assignMMT(outGrad, m_weights);
		}
		else if(input.dims() == 1)
		{
			NNAssertEquals(outGrad.shape(), m_outputShape, "Incompatible outGrad!");
			NNAssertEquals(input.shape(), m_inputShape, "Incompatible input!");
			m_inGrad.resize(input.size(0), m_inputShape[0]);
			
			// biasGrad (outs x 1) += outGrad^T (outs x 1) x addBuffer (1 x 1)
			m_biasGrad.addV(outGrad);
			
			// weightsGrad (inps x outs) += input (inps) x outGrad (outs)
			m_weightsGrad.assignVV(input, outGrad, 1, 1);
			
			// inGrad (inps) = weights (inps x outs) x outGrad (outs)
			m_inGrad.assignMTV(m_weights, outGrad);
		}
		else
		{
			throw Error("Incompatible input!");
		}
		
		return m_inGrad;
	}
	
	// MARK: Size Management
	
	/// Set the output shape of this module, including batch.
	virtual void resizeOutputs(const Storage<size_t> &dims) override
	{
		NNAssertEquals(dims.size(), 1, "Expected one-dimensional output!");
		Module<T>::resizeOutputs(dims);
		m_weights.resizeDim(1, dims[0]);
		m_weightsGrad.resizeDim(1, dims[0]);
		m_bias.resize(dims[0]);
		m_biasGrad.resize(dims[0]);
		reset();
	}
	
	/// Set the input shape of this module, including batch.
	virtual void resizeInputs(const Storage<size_t> &dims) override
	{
		NNAssertEquals(dims.size(), 1, "Expected one-dimensional input!");
		Module<T>::resizeInputs(dims);
		m_weights.resizeDim(0, dims[0]);
		m_weightsGrad.resizeDim(0, dims[0]);
		reset();
	}
	
	// MARK: Other Methods
	
	/// A vector of tensors filled with (views of) this module's parameters.
	virtual Storage<Tensor<T> *> parameterList() override
	{
		return { &m_weights, &m_bias };
	}
	
	/// A vector of tensors filled with (views of) this module's parameters' gradient.
	virtual Storage<Tensor<T> *> gradList() override
	{
		return { &m_weightsGrad, &m_biasGrad };
	}
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
	using Module<T>::m_outputShape;
	using Module<T>::m_inputShape;
	
private:
	Tensor<T> m_weights;		///< Module weights.
	Tensor<T> m_weightsGrad;	///< Gradient of the error w.r.t. the weights.
	Tensor<T> m_bias;			///< Network bias.
	Tensor<T> m_biasGrad;		///< Gradient of the error w.r.t. the bias.
	Tensor<T> m_addBuffer;		///< A vector of 1s for outer-producting bias.
};

}

NNRegisterType(Linear, Module);

#endif
