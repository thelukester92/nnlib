#ifndef NN_BATCHNORM_H
#define NN_BATCHNORM_H

#include "module.h"

namespace nnlib
{

template <typename T = double>
class BatchNorm : public Module<T>
{
public:
	using Module<T>::inputs;
	using Module<T>::outputs;
	
	BatchNorm(size_t inps = 0, size_t bats = 1) :
		m_output(bats, inps),
		m_inGrad(bats, inps),
		m_means(inps),
		m_variances(inps),
		m_params({ 1, 0 }),
		m_grads(2)
	{}
	
	T scale() const
	{
		return m_params(0);
	}
	
	BatchNorm &scale(T alpha)
	{
		m_params(0) = alpha;
		return *this;
	}
	
	T shift() const
	{
		return m_params(1);
	}
	
	BatchNorm &shift(T alpha)
	{
		m_params(1) = alpha;
		return *this;
	}
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		NNAssert(input.shape() == m_inGrad.shape(), "Incompatible input!");
		
		// Iterate over each column
		for(size_t i = 0, n = m_means.size(0); i < n; ++i)
		{
			// Get views of this column (input and output)
			const Tensor<T> &columnIn = input.select(1, i);
			Tensor<T> columnOut = m_output.select(1, i);
			
			// Calculate the mean of this column
			m_means(i) = columnIn.mean();
			
			// Calulate the variance of this column
			m_variances(i) = 0;
			for(const T &v : columnIn)
			{
				T diff = v - m_means(i);
				m_variances(i) += diff * diff;
			}
			m_variances(i) /= m_inGrad.size(0);
			
			// Normalize this column
			columnOut.copy(columnIn).shift(-m_means(i)).scale(1.0 / sqrt(m_variances(i) + 1e-12));
		}
		
		// Rescale and reshift the entire result
		m_output.scale(m_params(0)).shift(m_params(1));
		
		return m_output;
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssert(input.shape() == m_inGrad.shape(), "Incompatible input!");
		NNAssert(outGrad.shape() == m_output.shape(), "Incompatible outGrad!");
		
		// Unscale the entire result
		Tensor<T> innerGrad(m_inGrad.shape(), true);
		innerGrad.copy(outGrad).scale(m_params(0));
		
		// Iterate over each column
		for(size_t i = 0, n = m_means.size(0); i < n; ++i)
		{
			// Get views of this column (input, output, inGrad, and outGrad)
			const Tensor<T> &columnIn = input.select(1, i);
			const Tensor<T> &columnGradOut = outGrad.select(1, i);
			Tensor<T> columnOut = m_output.select(1, i);
			Tensor<T> columnGradIn = m_inGrad.select(1, i);
			
			Tensor<T> shiftedStuff = columnIn.copy().shift(-m_means(i));
			
			// Get gradient of variance(i)
			T varianceGrad = innerGrad.select(1, i).copy()
				.pointwiseProduct(shiftedStuff)
				.sum() * -0.5 * pow(m_variances(i) + 1e-12, -1.5);
			
			// Get gradient of mean(i)
			T meanGrad = innerGrad.select(1, i)
				.sum() * -1.0 / sqrt(m_variances(i) + 1e-12)
				+ varianceGrad * -2 * shiftedStuff.mean();
			
			// Get gradient of input
			columnGradIn.copy(innerGrad)
				.scale(1.0 / sqrt(m_variances(i) + 1e-12))
				.addVV(shiftedStuff.scale(varianceGrad * 2 / m_inGrad.size(0)))
				.shift(meanGrad / m_inGrad.size(0));
			
			// Get gradient of parameters
			/// \todo make this more efficient, I hacked it together at 5:05pm
			m_grads(0) += columnGradOut.copy().pointwiseProduct(columnOut.copy().shift(-m_grads(1)).scale(1.0 / m_grads(0))).sum();
			m_grads(1) += columnGradOut.sum();
		}
		
		return m_inGrad;
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
	
	/// Set the input shape of this module, including batch.
	/// In batchnorm, input shape is always equal to output shape.
	virtual BatchNorm &inputs(const Storage<size_t> &dims) override
	{
		NNAssert(dims.size() == 2, "BatchNorm expects matrix inputs!");
		Module<T>::inputs(dims);
		Module<T>::outputs(dims);
		m_means.resize(dims[1]);
		m_variances.resize(dims[1]);
		return *this;
	}
	
	/// Set the output shape of this module, including batch.
	/// In batchnorm, input shape is always equal to output shape.
	virtual BatchNorm &outputs(const Storage<size_t> &dims) override
	{
		return inputs(dims);
	}
	
	/// A vector of tensors filled with (views of) this module's parameters.
	virtual Storage<Tensor<T> *> parameterList() override
	{
		return { &m_params };
	}
	
	/// A vector of tensors filled with (views of) this module's parameters' gradient.
	virtual Storage<Tensor<T> *> gradList() override
	{
		return { &m_grads };
	}
	
	/// A vector of tensors filled with (views of) this module's internal state.
	virtual Storage<Tensor<T> *> stateList() override
	{
		Storage<Tensor<T> *> states = Module<T>::stateList();
		states.push_back(&m_means);
		states.push_back(&m_variances);
		return states;
	}
	
	// MARK: Serialization
	
	/// \brief Write to an archive.
	///
	/// The archive takes care of whitespace for plaintext.
	/// \param out The archive to which to write.
	virtual void save(Archive &out) const override
	{
		out << Binding<BatchNorm>::name << m_params(0) << m_params(1);
	}
	
	/// \brief Read from an archive.
	///
	/// \param in The archive from which to read.
	virtual void load(Archive &in) override
	{
		std::string str;
		in >> str;
		NNAssert(
			str == Binding<BatchNorm>::name,
			"Unexpected type! Expected '" + Binding<BatchNorm>::name + "', got '" + str + "'!"
		);
		in >> m_params(0) >> m_params(1);
	}
private:
	Tensor<T> m_output;
	Tensor<T> m_inGrad;
	Tensor<T> m_means;
	Tensor<T> m_variances;
	Tensor<T> m_params;	///< A two-element tensor continaing scale and shift.
	Tensor<T> m_grads;	///< Gradient of error w.r.t. m_params.
};

NNSerializable(BatchNorm<double>, Module<double>);
NNSerializable(BatchNorm<float>, Module<float>);

}

#endif
