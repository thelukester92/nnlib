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
		m_invStds(inps),
		m_runningMeans(inps),
		m_runningVars(inps),
		m_normalized(bats, inps),
		m_weights(inps),
		m_biases(inps),
		m_weightsGrad(inps),
		m_biasesGrad(inps),
		m_training(true),
		m_momentum(0.1)
	{
		m_weights.fill(1);
	}
	
	/// Returns whether this module is in training mode.
	bool training() const
	{
		return m_training;
	}
	
	/// Sets whether this module is in training mode.
	BatchNorm &training(bool training)
	{
		m_training = training;
		return *this;
	}
	
	/// Returns this module's momentum for running averages.
	T momentum() const
	{
		return m_momentum;
	}
	
	/// Sets this module's momentum for running averages.
	BatchNorm &momentum(T momentum)
	{
		m_momentum = momentum;
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
			Tensor<T> columnNorm = m_normalized.select(1, i);
			Tensor<T> columnOut = m_output.select(1, i);
			
			// Get mean and variance to use
			T mean, invstd;
			
			if(m_training)
			{
				// Calculate the mean of this column
				mean = columnIn.mean();
				m_means(i) = mean;
				
				// Calulate the inverted standard deviation of this column
				T sum = 0;
				for(const T &v : columnIn)
				{
					T diff = v - mean;
					sum += diff * diff;
				}
				invstd = 1.0 / sqrt(sum / m_inGrad.size(0) + 1e-12);
				m_invStds(i) = invstd;
				
				// Update running mean
				m_runningMeans(i) = m_momentum * mean + (1 - m_momentum) * m_runningMeans(i);
				
				// Update running variance (unbiased)
				m_runningVars(i) = m_momentum * sum / (m_inGrad.size() - 1) + (1 - m_momentum) * m_runningVars(i);
			}
			else
			{
				mean = m_runningMeans(i);
				invstd = 1.0 / sqrt(m_runningVars(i) + 1e-12);
			}
			
			// Normalize this column
			columnNorm.copy(columnIn).shift(-mean).scale(invstd);
			
			// Shift and scale using parameters
			columnOut.copy(columnNorm).scale(m_weights(i)).shift(m_biases(i));
		}
		
		return m_output;
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssert(input.shape() == m_inGrad.shape(), "Incompatible input!");
		NNAssert(outGrad.shape() == m_output.shape(), "Incompatible outGrad!");
		
		/*
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
			Tensor<T> columnNorm = m_normalized.select(1, i);
			Tensor<T> shiftedStuff = columnIn.copy().shift(-m_means(i));
			
			// Get gradient of variance(i)
			T varianceGrad = innerGrad.select(1, i).copy()
				.pointwiseProduct(shiftedStuff)
				.sum() * -0.5 * pow(m_invStds(i) + 1e-12, -1.5);
			
			// Get gradient of mean(i)
			T meanGrad = innerGrad.select(1, i)
				.sum() * -1.0 / sqrt(m_invStds(i) + 1e-12)
				+ varianceGrad * -2 * shiftedStuff.mean();
			
			// Get gradient of input
			columnGradIn.copy(innerGrad)
				.scale(1.0 / sqrt(m_invStds(i) + 1e-12))
				.addVV(shiftedStuff.scale(varianceGrad * 2 / m_inGrad.size(0)))
				.shift(meanGrad / m_inGrad.size(0));
			
			// Get gradient of parameters
			m_grads(0) += columnGradOut.copy().pointwiseProduct(columnNorm).sum();
			m_grads(1) += columnGradOut.sum();
		}
		*/
		
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
	
	/// Set the input and output shapes of this module.
	/// In batchnorm, input shape is always equal to output shape.
	virtual BatchNorm &resize(const Storage<size_t> &inps, const Storage<size_t> &outs) override
	{
		NNAssert(inps == outs, "BatchNorm expects the same input and output size!");
		return inputs(outs);
	}
	
	/// Safely (never reset weights) set the input and output shapes of this module.
	/// In batchnorm, input shape is always equal to output shape.
	virtual BatchNorm &safeResize(const Storage<size_t> &inps, const Storage<size_t> &outs) override
	{
		NNAssert(inps == outs, "BatchNorm expects the same input and output size!");
		this->safeInputs(inps);
		return *this;
	}
	
	/// Set the input shape of this module, including batch.
	/// In batchnorm, input shape is always equal to output shape.
	virtual BatchNorm &inputs(const Storage<size_t> &dims) override
	{
		NNAssert(dims.size() == 2, "BatchNorm expects matrix inputs!");
		Module<T>::inputs(dims);
		Module<T>::outputs(dims);
		m_means.resize(dims[1]);
		m_invStds.resize(dims[1]);
		m_runningMeans.resize(dims[1]);
		m_runningVars.resize(dims[1]);
		m_normalized.resize(dims);
		m_weights.resize(dims[1]).fill(1);
		m_biases.resize(dims[1]);
		m_weightsGrad.resize(dims[1]);
		m_biasesGrad.resize(dims[1]);
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
		return { &m_weights, &m_biases };
	}
	
	/// A vector of tensors filled with (views of) this module's parameters' gradient.
	virtual Storage<Tensor<T> *> gradList() override
	{
		return { &m_weightsGrad, &m_biasesGrad };
	}
	
	/// A vector of tensors filled with (views of) this module's internal state.
	virtual Storage<Tensor<T> *> stateList() override
	{
		Storage<Tensor<T> *> states = Module<T>::stateList();
		states.push_back(&m_means);
		states.push_back(&m_invStds);
		states.push_back(&m_normalized);
		return states;
	}
	
	// MARK: Serialization
	
	/// \brief Write to an archive.
	///
	/// The archive takes care of whitespace for plaintext.
	/// \param out The archive to which to write.
	virtual void save(Archive &out) const override
	{
		out << Binding<BatchNorm>::name << m_means << m_invStds << m_training << m_momentum;
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
		
		in >> m_means >> m_invStds;
		NNAssert(
			m_means.size() == m_invStds.size() && m_means.dims() == 1 && m_invStds.dims() == 1,
			"Incompatible means and variances!"
		);
		inputs({ m_inGrad.size(0), m_means.size() });
		
		in >> m_training >> m_momentum;
	}
private:
	Tensor<T> m_output;				///< Cached output.
	Tensor<T> m_inGrad;				///< Gradient of error w.r.t. inputs.
	Tensor<T> m_means;				///< Mean of each input dimension within the batch.
	Tensor<T> m_invStds;			///< Inverted standard deviation of each input dimension within the batch.
	Tensor<T> m_runningMeans;		///< A running mean for each input, for after training.
	Tensor<T> m_runningVars;	///< A running variance for each input, for after training.
	Tensor<T> m_normalized;			///< Halfway transformed input, cached for backward.
	Tensor<T> m_weights;			///< How much to scale the normalized input.
	Tensor<T> m_biases;				///< How much to shift the normalized input.
	Tensor<T> m_weightsGrad;		///< Gradient of error w.r.t. weights.
	Tensor<T> m_biasesGrad;			///< Gradient of error w.r.t. biases.
	bool m_training;				///< Whether we are in training mode.
	T m_momentum;					///< How much to update running mean and variance.
};

NNSerializable(BatchNorm<double>, Module<double>);
NNSerializable(BatchNorm<float>, Module<float>);

}

#endif
