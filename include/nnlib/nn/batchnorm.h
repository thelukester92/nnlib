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
		size_t n = input.size(0);
		
		// Get means and variances to use
		Tensor<T> means, invStds;
		if(m_training)
		{
			T norm = 1.0 / n;
			
			// Get means
			input.sum(m_means, 0);
			m_means.scale(norm);
			
			// Get unnormalized variances (temporarily stored in invStd)
			for(size_t i = 0; i < n; ++i)
			{
				m_invStds.addV(input.select(0, i).copy().addV(m_means, -1).square());
			}
			
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
		NNAssert(input.shape() == m_inGrad.shape(), "Incompatible input!");
		NNAssert(outGrad.shape() == m_output.shape(), "Incompatible outGrad!");
		size_t n = input.size(0);
		
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
			{
				invStd = 1.0 / sqrt(invStd + 1e-12);
			}
		}
		
		// gradient of biases
		m_biasesGrad.addV(outGrad.sum(0));
		
		// gradient of weights
		Tensor<T> product = m_normalized.copy().pointwiseProduct(outGrad);
		for(size_t i = 0; i < n; ++i)
		{
			m_weightsGrad.addV(product.select(0, i));
		}
		
		// gradient of inputs
		if(m_training)
		{
			Tensor<T> gradMean = outGrad.sum(0).scale(1.0 / n);
			m_inGrad.copy(outGrad);
			
			Tensor<T> stuff = m_normalized.pointwiseProduct(outGrad).sum(0);
			
			for(size_t i = 0; i < n; ++i)
			{
				m_inGrad.select(0, i)
					.add(gradMean, -1)
					.add(
						input.select(0, i).copy()
						.add(means, -1)
						.pointwiseProduct(stuff)
						.pointwiseProduct(invStds).scale(1.0 / n), -1)
					.pointwiseProduct(invStds)
					.pointwiseProduct(m_weights);
			}
		}
		else
		{
			m_inGrad.copy(outGrad);
			for(size_t i = 0; i < n; ++i)
			{
				m_inGrad.select(0, i).pointwiseProduct(invStds).pointwiseProduct(m_weights);
			}
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
	
	// MARK: Serialization
	/*
	/// \brief Write to an archive.
	///
	/// The archive takes care of whitespace for plaintext.
	/// \param out The archive to which to write.
	virtual void save(Archive &out) const override
	{
		out << Binding<BatchNorm>::name << m_runningMeans << m_runningVars << m_weights << m_biases << m_training << m_momentum << m_inGrad.size(0);
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
		
		size_t bats;
		in >> m_runningMeans >> m_runningVars >> m_weights >> m_biases >> m_training >> m_momentum >> bats;
		NNAssert(
			m_runningMeans.shape() == m_runningVars.shape()
			&& m_runningMeans.shape() == m_weights.shape()
			&& m_runningMeans.shape() == m_biases.shape()
			&& m_runningMeans.dims() == 1,
			"Incompatible means and variances!"
		);
		inputs({ bats, m_runningMeans.size(0) });
	}
	*/
	
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

NNSerializable(BatchNorm<double>, Module<double>);
NNSerializable(BatchNorm<float>, Module<float>);

}

#endif
