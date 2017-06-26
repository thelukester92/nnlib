#ifndef DROPOUT_H
#define DROPOUT_H

#include "module.h"

namespace nnlib
{

template <typename T = double>
class Dropout : public Module<T>
{
	using Module<T>::m_training;
public:
	using Module<T>::inputs;
	using Module<T>::outputs;
	
	Dropout(T dropProbability = 0.1, size_t inps = 0, size_t bats = 1) :
		m_inGrad(bats, inps),
		m_output(bats, inps),
		m_mask(bats, inps),
		m_dropProbability(dropProbability)
	{}
	
	Dropout(const Dropout &module) :
		m_inGrad(module.m_inGrad.copy()),
		m_output(module.m_output.copy()),
		m_mask(module.m_mask.copy()),
		m_dropProbability(module.m_dropProbability)
	{
		m_training = module.m_training;
	}
	
	Dropout &operator=(const Dropout &module)
	{
		m_inGrad	= module.m_inGrad.copy();
		m_output	= module.m_output.copy();
		m_training	= module.m_training;
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
		m_dropProbability = dropProbability;
		return *this;
	}
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		NNAssertEquals(input.shape(), m_inGrad.shape(), "Incompatible input!");
		if(m_training)
			return m_output.copy(input).pointwiseProduct(m_mask.bernoulli(1 - m_dropProbability));
		else
			return m_output.copy(input).scale(1 - m_dropProbability);
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssertEquals(input.shape(), m_inGrad.shape(), "Incompatible input!");
		NNAssertEquals(outGrad.shape(), m_output.shape(), "Incompatible output!");
		if(m_training)
			return m_inGrad.copy(outGrad).pointwiseProduct(m_mask);
		else
			return m_inGrad.copy(outGrad).scale(1 - m_dropProbability);
	}
	
	/// Cached output.
	virtual Tensor<T> &output() override
	{
		return m_output;
	}
	
	/// Cached input gradient.
	virtual Tensor<T> &inGrad() override
	{
		return m_inGrad;
	}
	
	/// Set the input and output shapes of this module.
	/// In dropout, input shape is always equal to output shape.
	virtual Dropout &resize(const Storage<size_t> &inps, const Storage<size_t> &outs) override
	{
		NNAssertEquals(inps, outs, "Expected input and output sizes to be equal!");
		return inputs(outs);
	}
	
	/// Safely (never reset weights) set the input and output shapes of this module.
	/// In dropout, input shape is always equal to output shape.
	virtual Dropout &safeResize(const Storage<size_t> &inps, const Storage<size_t> &outs) override
	{
		NNAssertEquals(inps, outs, "Expected input and output sizes to be equal!");
		this->safeInputs(inps);
		return *this;
	}
	
	/// Set the input shape of this module, including batch.
	/// In dropout, input shape is always equal to output shape.
	virtual Dropout &inputs(const Storage<size_t> &dims) override
	{
		NNAssertEquals(dims.size(), 2, "Expected matrix input!");
		Module<T>::inputs(dims);
		Module<T>::outputs(dims);
		return *this;
	}
	
	/// Set the output shape of this module, including batch.
	/// In dropout, input shape is always equal to output shape.
	virtual Dropout &outputs(const Storage<size_t> &dims) override
	{
		return inputs(dims);
	}
	
	/// A vector of tensors filled with (views of) this module's internal state.
	virtual Storage<Tensor<T> *> stateList() override
	{
		Storage<Tensor<T> *> states = Module<T>::stateList();
		states.push_back(&m_mask);
		return states;
	}
	
	/// \brief Write to an archive.
	///
	/// The archive takes care of whitespace for plaintext.
	/// \param ar The archive to which to write.
	template <typename Archive>
	void save(Archive &ar) const
	{
		ar(this->inputs());
	}
	
	/// \brief Read from an archive.
	///
	/// \param ar The archive from which to read.
	template <typename Archive>
	void load(Archive &ar)
	{
		Storage<size_t> shape;
		ar(shape);
		this->inputs(shape);
	}
	
private:
	Tensor<T> m_inGrad;		///< Input gradient buffer.
	Tensor<T> m_output;		///< Output buffer.
	Tensor<T> m_mask;		///< Randomly-generated mask.
	T m_dropProbability;	///< The probability that an output is dropped.
};

}

NNRegisterType(Dropout<float>, Module<float>);
NNRegisterType(Dropout<double>, Module<double>);

#endif
