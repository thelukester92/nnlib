#ifndef NN_BATCHNORM_HPP
#define NN_BATCHNORM_HPP

#include "module.hpp"

namespace nnlib
{

/// \brief Batch normalization.
///
/// Works only on 2D tensors.
template <typename T = double>
class BatchNorm : public Module<T>
{
public:
	BatchNorm(size_t inps) :
		m_weights(inps),
		m_weightsGrad(inps),
		m_biases(inps),
		m_biasesGrad(inps),
		m_runningMeans(inps),
		m_runningVars(inps),
		m_means(inps),
		m_invStds(inps),
		m_momentum(0.1),
		m_training(true)
	{
		reset();
	}
	
	BatchNorm(const BatchNorm &module) :
		m_weights(module.m_weights.copy()),
		m_weightsGrad(module.m_weightsGrad.copy()),
		m_biases(module.m_biases.copy()),
		m_biasesGrad(module.m_biasesGrad.copy()),
		m_runningMeans(module.m_runningMeans.copy()),
		m_runningVars(module.m_runningVars.copy()),
		m_means(module.m_means.copy()),
		m_invStds(module.m_invStds.copy()),
		m_momentum(module.m_momentum),
		m_training(module.m_training)
	{}
	
	BatchNorm(const Serialized &node) :
		m_weights(node.get<Tensor<T>>("weights")),
		m_weightsGrad(m_weights.shape(), true),
		m_biases(node.get<Tensor<T>>("biases")),
		m_biasesGrad(m_biases.shape(), true),
		m_runningMeans(node.get<Tensor<T>>("runningMeans")),
		m_runningVars(node.get<Tensor<T>>("runningVars")),
		m_means(m_weights.shape(), true),
		m_invStds(m_weights.shape(), true),
		m_momentum(node.get<T>("momentum")),
		m_training(node.get<bool>("training"))
	{
		NNAssertEquals(m_biases.shape(), m_weights.shape(), "Incompatible bias!");
		NNAssertEquals(m_runningMeans.shape(), m_weights.shape(), "Incompatible running means!");
		NNAssertEquals(m_runningVars.shape(), m_weights.shape(), "Incompatible running variances!");
	}
	
	BatchNorm &operator=(BatchNorm module)
	{
		swap(*this, module);
		return *this;
	}
	
	friend void swap(BatchNorm &a, BatchNorm &b)
	{
		using std::swap;
		swap(a.m_means, b.m_means);
		swap(a.m_invStds, b.m_invStds);
		swap(a.m_runningMeans, b.m_runningMeans);
		swap(a.m_runningVars, b.m_runningVars);
		swap(a.m_weights, b.m_weights);
		swap(a.m_biases, b.m_biases);
		swap(a.m_weightsGrad, b.m_weightsGrad);
		swap(a.m_biasesGrad, b.m_biasesGrad);
		swap(a.m_momentum, b.m_momentum);
		swap(a.m_training, b.m_training);
	}
	
	/// Resets the weights/biases.
	BatchNorm &reset()
	{
		if(m_weights.size(0) > 0)
		{
			m_weights.rand();
			m_biases.zeros();
		}
		return *this;
	}
	
	/// Get the weights of this module.
	Tensor<T> weights()
	{
		return m_weights;
	}
	
	/// Get the bias of this module.
	Tensor<T> bias()
	{
		return m_biases;
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
		node.set("weights", m_weights);
		node.set("biases", m_biases);
		node.set("runningMeans", m_runningMeans);
		node.set("runningVars", m_runningVars);
		node.set("momentum", m_momentum);
		node.set("training", m_training);
	}
	
	// MARK: Computation
	
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		NNAssertEquals(input.dims(), 2, "Expected matrix input!");
		NNAssertEquals(input.size(1), m_weights.size(), "Incompatible input!");
		m_output.resize(input.shape());
		
		size_t n = input.size(0);
		T norm = 1.0 / n;
		
		// Get means and variances to use
		Tensor<T> means, invStds;
		if(m_training)
		{
			NNAssertGreaterThan(input.size(0), 1, "Expected a batch in training mode!");
			
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
		forEach([&](T &invStd)
		{
			invStd = 1.0 / sqrt(invStd + 1e-12);
		}, invStds);
		
		// Use the statistics to normalize the data row-by-row
		for(size_t i = 0; i < n; ++i)
		{
			Tensor<T> out = m_output.select(0, i);
			
			// Copy
			out.copy(input.select(0, i));
			
			// Normalize
			out.addV(means, -1).pointwiseProduct(invStds);
			
			// Rescale and reshift using the parameters
			out.pointwiseProduct(m_weights).addV(m_biases);
		}
		
		return m_output;
	}
	
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssertEquals(input.dims(), 2, "Expected matrix input!");
		NNAssertEquals(input.size(1), m_weights.size(), "Incompatible input!");
		NNAssertEquals(input.shape(), outGrad.shape(), "Incompatible outGrad!");
		m_output.resize(input.shape());
		m_inGrad.resize(input.shape());
		
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
			forEach([&](T &invStd)
			{
				invStd = 1.0 / sqrt(invStd + 1e-12);
			}, invStds);
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
			m_biasesGrad(i) += sum;
			
			// gradient of weights
			m_weightsGrad(i) += dotp * invStds(i);
		}
		
		return m_inGrad;
	}
	
	// MARK: Buffers
	
	/// A vector of tensors filled with (views of) this module's parameters.
	virtual Storage<Tensor<T> *> paramsList() override
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
		return Module<T>::stateList().append({ &m_means, &m_invStds, &m_runningMeans, &m_runningVars });
	}
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
	
private:
	Tensor<T> m_weights;		///< How much to scale the normalized input.
	Tensor<T> m_weightsGrad;	///< Gradient of error w.r.t. weights.
	Tensor<T> m_biases;			///< How much to shift the normalized input.
	Tensor<T> m_biasesGrad;		///< Gradient of error w.r.t. biases.
	
	Tensor<T> m_runningMeans;	///< A running mean for each input, for after training.
	Tensor<T> m_runningVars;	///< A running variance for each input, for after training.
	
	Tensor<T> m_means;			///< Mean of each input dimension within the batch.
	Tensor<T> m_invStds;		///< Inverted standard deviation of each input dimension within the batch.
	
	T m_momentum;				///< How much to update running mean and variance.
	bool m_training;			///< Whether this module is in training or evaluation mode.
};

}

NNRegisterType(BatchNorm, Module);

#endif
