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
	using Module<T>::batch;
	
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
		m_biases.fill(0);
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
		NNAssertEquals(input.shape(), m_inGrad.shape(), "Incompatible input!");
		size_t n = input.size(0);
		T norm = 1.0 / n;
		
		// Get means and variances to use
		Tensor<T> means, invStds;
		if(m_training)
		{
			// Get means
			input.sum(m_means, 0);
			m_means.scale(norm);
			
			// Get unnormalized variances (temporarily stored in m_invStds)
			for(size_t i = 0; i < n; ++i)
				m_invStds.addV(input.select(0, i).copy().addV(m_means, -1).square());
			
			// Update running mean
			m_runningMeans.scale(1 - m_momentum).addV(m_means.copy().scale(m_momentum));
			
			// Update running variance (normalize as sample)
			m_runningVars.scale(1 - m_momentum).addV(m_invStds.copy().scale(m_momentum / (n - 1)));
			
			// Now normalize variance as population; will invert and sqrt after this if statement
			m_invStds.scale(norm);
			
			// Use the batch statistics
			means = m_means;
			invStds = m_invStds;
		}
		else
		{
			// Use the running statistics
			means = m_runningMeans;
			invStds = m_runningVars.copy();
		}
		
		// Turn variance into inverted standard deviation
		for(T &invStd : invStds)
		{
			invStd = 1.0 / sqrt(invStd + 1e-12);
		}
		
		// Use the statistics to normalize the data row-by-row
		m_normalized.copy(input);
		for(size_t i = 0; i < n; ++i)
		{
			Tensor<T> nrm = m_normalized.select(0, i);
			Tensor<T> out = m_output.select(0, i);
			
			// Normalize
			nrm.addV(means, -1).pointwiseProduct(invStds);
			
			// Rescale and reshift using the parameters
			out.copy(nrm).pointwiseProduct(m_weights).addV(m_biases);
		}
		
		return m_output;
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssertEquals(input.shape(), m_inGrad.shape(), "Incompatible input!");
		NNAssertEquals(outGrad.shape(), m_output.shape(), "Incompatible output!");
		size_t inps = m_inGrad.size(1), bats = m_inGrad.size(0);
		
		// Get means and variances to use
		Tensor<T> means, invStds;
		if(m_training)
		{
			// Use the batch statistics
			means = m_means;
			invStds = m_invStds;
		}
		else
		{
			// Use the running statistics
			means = m_runningMeans;
			invStds = m_runningVars.copy();
			
			// Turn variance into inverted standard deviation
			for(T &invStd : invStds)
				invStd = 1.0 / sqrt(invStd + 1e-12);
		}
		
		for(size_t i = 0; i < inps; ++i)
		{
			T sum = 0;
			for(size_t j = 0; j < bats; ++j)
				sum += outGrad(j, i);
			
			T dotp = 0;
			for(size_t j = 0; j < bats; ++j)
				dotp += (input(j, i) - means(i)) * outGrad(j, i);
			
			// gradient of inputs
			if(m_training)
			{
				T k = dotp * invStds(i) * invStds(i) / bats;
				for(size_t j = 0; j < bats; ++j)
					m_inGrad(j, i) = (input(j, i) - means(i)) * k;
				
				T gradMean = sum / bats;
				for(size_t j = 0; j < bats; ++j)
					m_inGrad(j, i) = (outGrad(j, i) - gradMean - m_inGrad(j, i)) * invStds(i) * m_weights(i);
			}
			else
			{
				for(size_t j = 0; j < bats; ++j)
					m_inGrad(j, i) = outGrad(j, i) * invStds(i) * m_weights(i);
			}
		
			// gradient of biases
			m_biasesGrad(i) += sum; // .addV(outGrad.sum(0));
			
			// gradient of weights
			m_weightsGrad(i) += dotp * invStds(i);
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
	
	/// Set the input and output shapes of this module.
	/// In batchnorm, input shape is always equal to output shape.
	virtual BatchNorm &resize(const Storage<size_t> &inps, const Storage<size_t> &outs) override
	{
		NNAssertEquals(inps, outs, "Expected input and output sizes to be equal!");
		return inputs(outs);
	}
	
	/// Safely (never reset weights) set the input and output shapes of this module.
	/// In batchnorm, input shape is always equal to output shape.
	virtual BatchNorm &safeResize(const Storage<size_t> &inps, const Storage<size_t> &outs) override
	{
		NNAssertEquals(inps, outs, "Expected input and output sizes to be equal!");
		this->safeInputs(inps);
		return *this;
	}
	
	/// Set the input shape of this module, including batch.
	/// In batchnorm, input shape is always equal to output shape.
	virtual BatchNorm &inputs(const Storage<size_t> &dims) override
	{
		NNAssertEquals(dims.size(), 2, "Expected matrix input!");
		Module<T>::inputs(dims);
		Module<T>::outputs(dims);
		m_means.resize(dims[1]);
		m_invStds.resize(dims[1]);
		m_runningMeans.resize(dims[1]);
		m_runningVars.resize(dims[1]);
		m_normalized.resize(dims);
		m_weights.resize(dims[1]).fill(1);
		m_biases.resize(dims[1]).fill(0);
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
	
	/// Set the batch size of this module.
	virtual BatchNorm &batch(size_t bats) override
	{
		Module<T>::batch(bats);
		m_normalized.resizeDim(0, bats);
		return *this;
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
		states.push_back(&m_runningMeans);
		states.push_back(&m_runningVars);
		return states;
	}
	
	/// \brief Write to an archive.
	///
	/// The archive takes care of whitespace for plaintext.
	/// \param ar The archive to which to write.
	template <typename Archive>
	void save(Archive &ar) const
	{
		ar(m_runningMeans, m_runningVars, m_weights, m_biases, m_training, m_momentum, m_inGrad.size(0));
	}
	
	/// \brief Read from an archive.
	///
	/// \param ar The archive from which to read.
	template <typename Archive>
	void load(Archive &ar)
	{
		size_t bats;
		ar(m_runningMeans, m_runningVars, m_weights, m_biases, m_training, m_momentum, bats);
		
		NNAssertEquals(m_runningMeans.shape(), m_runningVars.shape(), "Incompatible means and variances!");
		NNAssertEquals(m_runningMeans.shape(), m_weights.shape(), "Incompatible means and weights!");
		NNAssertEquals(m_runningMeans.shape(), m_biases.shape(), "Incompatible means and biases!");
		NNAssertEquals(m_runningMeans.dims(), 1, "Expected means to be a vector!");
		
		inputs({ bats, m_runningMeans.size(0) });
	}
	
private:
	Tensor<T> m_output;			///< Cached output.
	Tensor<T> m_inGrad;			///< Gradient of error w.r.t. inputs.
	Tensor<T> m_means;			///< Mean of each input dimension within the batch.
	Tensor<T> m_invStds;		///< Inverted standard deviation of each input dimension within the batch.
	Tensor<T> m_runningMeans;	///< A running mean for each input, for after training.
	Tensor<T> m_runningVars;	///< A running variance for each input, for after training.
	Tensor<T> m_normalized;		///< Halfway transformed input, cached for backward.
	Tensor<T> m_weights;		///< How much to scale the normalized input.
	Tensor<T> m_biases;			///< How much to shift the normalized input.
	Tensor<T> m_weightsGrad;	///< Gradient of error w.r.t. weights.
	Tensor<T> m_biasesGrad;		///< Gradient of error w.r.t. biases.
	bool m_training;			///< Whether we are in training mode.
	T m_momentum;				///< How much to update running mean and variance.
};

}

NNRegisterType(BatchNorm<double>, Module<double>);
NNRegisterType(BatchNorm<float>, Module<float>);

#endif
