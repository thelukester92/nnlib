#ifndef NN_LINEAR_H
#define NN_LINEAR_H

#include "module.h"

namespace nnlib
{

/// A standard feed-forward layer that returns a linear combination of inputs.
class Linear : public Module
{
public:
	using Module::inputs;
	using Module::outputs;
	using Module::batch;
	
	/// \brief A name for this module type.
	///
	/// This may be used for debugging, serialization, etc.
	/// The type should NOT include whitespace.
	static std::string type()
	{
		return "linear";
	}
	
	/// Standard inps -> outs layer.
	Linear(size_t inps, size_t outs, size_t bats = 1) :
		m_weights(inps, outs),
		m_weightsGrad(inps, outs),
		m_bias(outs),
		m_biasGrad(outs),
		m_inGrad(bats, inps),
		m_output(bats, outs),
		m_addBuffer(bats)
	{
		reset();
	}
	
	/// any -> outs layer; adding to a sequential will set input size.
	Linear(size_t outs = 0) :
		m_weights(0, outs),
		m_weightsGrad(0, outs),
		m_bias(outs),
		m_biasGrad(outs),
		m_inGrad(1, 0),
		m_output(1, outs),
		m_addBuffer(1)
	{}
	
	/// Set weights to uniformly distributed random values.
	Linear &reset()
	{
		if(m_weights.size() > 0)
		{
			real_t range = 1.0 / sqrt(m_weights.size(1));
			m_weights.rand(-range, range);
			m_bias.rand(-range, range);
		}
		return *this;
	}
	
	/// Get the weights of this module.
	Tensor &weights()
	{
		return m_weights;
	}
	
	/// Get the bias of this module.
	Tensor &bias()
	{
		return m_bias;
	}
	
	// MARK: Module methods
	
	/// Forward propagate input, returning output.
	virtual Tensor &forward(const Tensor &input) override
	{
		NNAssert(input.dims() == 2, "Linear expects Matrix input!");
		
		// output (bats x outs) = input (bats x inps) x weights (inps x outs)
		m_output.multiplyMM(input, m_weights);
		
		// output (bats x outs) += addBuffer (bats x 1) x bias (1 x outs)
		m_output.multiplyVTV(m_addBuffer, m_bias);
		
		return m_output;
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor &backward(const Tensor &input, const Tensor &outGrad) override
	{
		NNAssert(input.dims() == 2, "Linear expects Matrix input!");
		NNAssert(outGrad.dims() == 2, "Linear expects Matrix output gradient!");
		
		// biasGrad (outs x 1) += outGrad^T (outs x bats) x addBuffer (bats x 1)
		m_biasGrad.multiplyMTV(outGrad, m_addBuffer, 1, 1);
		
		// weightsGrad (inps x outs) += input^T (bats x inps) x outGrad (bats x outs)
		m_weightsGrad.multiplyMTM(input, outGrad, 1, 1);
		
		// inGrad (bats x inps) = outGrad (bats x outs) x weights^T (outs x inps)
		m_inGrad.multiplyMMT(outGrad, m_weights);
		
		return m_inGrad;
	}
	
	/// Cached output.
	virtual Tensor &output() override
	{
		return m_output;
	}
	
	/// Cached input gradient.
	virtual Tensor &inGrad() override
	{
		return m_inGrad;
	}
	
	/// Set the input shape of this module, including batch.
	virtual Linear &inputs(const Storage<size_t> &dims) override
	{
		NNAssert(dims.size() == 2, "Linear only works with matrix inputs!");
		Module::inputs(dims);
		m_weights.resize(m_inGrad.size(1), m_output.size(1));
		m_weightsGrad.resize(m_inGrad.size(1), m_output.size(1));
		return reset();
	}
	
	/// Set the output shape of this module, including batch.
	virtual Linear &outputs(const Storage<size_t> &dims) override
	{
		NNAssert(dims.size() == 2, "Linear only works with matrix outputs!");
		Module::outputs(dims);
		m_weights.resize(m_inGrad.size(1), m_output.size(1));
		m_weightsGrad.resize(m_inGrad.size(1), m_output.size(1));
		m_bias.resize(m_output.size(1));
		m_biasGrad.resize(m_output.size(1));
		return reset();
	}
	
	/// Set the batch size of this module.
	virtual Linear &batch(size_t bats) override
	{
		Module::batch(bats);
		m_addBuffer.resize(bats);
		return *this;
	}
	
	/// A vector of tensors filled with (views of) this module's parameters.
	virtual Storage<Tensor *> parameterList() override
	{
		return { &m_weights, &m_bias };
	}
	
	/// A vector of tensors filled with (views of) this module's parameters' gradient.
	virtual Storage<Tensor *> gradList() override
	{
		return { &m_weightsGrad, &m_biasGrad };
	}
	
	// MARK: Serialization
	
	/// \brief Write to an archive.
	///
	/// The archive takes care of whitespace for plaintext.
	/// \param out The archive to which to write.
	virtual void save(Archive &out) const override
	{
		out << type() << m_weights << m_bias << batch();
	}
	
	/// \brief Read from an archive.
	///
	/// \param in The archive from which to read.
	virtual void load(Archive &in) override
	{
		std::string str;
		in >> str;
		NNAssert(str == type(), "Unexpected type! Expected '" + type() + "', got '" + str + "'!");
		
		size_t bats;
		in >> m_weights >> m_bias >> bats;
		NNAssert(m_weights.dims() == 2 && m_bias.dims() == 1, "Unexpected tensor for Linear!");
		
		Module::inputs({ bats, m_weights.size(0) });
		Module::outputs({ bats, m_weights.size(1) });
		m_weightsGrad.resize(m_weights.shape());
		m_biasGrad.resize(m_bias.shape());
	}
	
private:
	Tensor m_weights;		///< Module weights.
	Tensor m_weightsGrad;	///< Gradient of the error w.r.t. the weights.
	
	Tensor m_bias;			///< Network bias.
	Tensor m_biasGrad;		///< Gradient of the error w.r.t. the bias.
	
	Tensor m_inGrad;			///< Input gradient buffer.
	Tensor m_output;			///< Output buffer.
	
	Tensor m_addBuffer;		///< A vector of 1s for outer-producting bias.
};

}

#endif
