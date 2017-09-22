#ifndef NN_BATCHNORM_H
#define NN_BATCHNORM_H

#include "module.h"

namespace nnlib
{

template <typename T = double>
class BatchNorm : public Module<T>
{
public:
	BatchNorm(size_t inps = 0, size_t bats = 1) :
		Module<T>({ bats, inps }, { bats, inps }),
		m_means(inps),
		m_invStds(inps),
		m_runningMeans(inps),
		m_runningVars(inps),
		m_normalized(bats, inps),
		m_weights(inps),
		m_biases(inps),
		m_weightsGrad(inps),
		m_biasesGrad(inps),
		m_momentum(0.1)
	{
		m_output.resize(bats, inps);
		m_inGrad.resize(bats, inps);
		reset();
	}
	
	BatchNorm(const BatchNorm &module) :
		Module<T>(module.inputShape(), module.outputShape()),
		m_means(module.m_means.copy()),
		m_invStds(module.m_invStds.copy()),
		m_runningMeans(module.m_runningMeans.copy()),
		m_runningVars(module.m_runningVars.copy()),
		m_normalized(module.m_normalized.copy()),
		m_weights(module.m_weights.copy()),
		m_biases(module.m_biases.copy()),
		m_weightsGrad(module.m_weightsGrad.copy()),
		m_biasesGrad(module.m_biasesGrad.copy()),
		m_momentum(module.m_momentum)
	{
		m_output = module.m_output.copy();
		m_inGrad = module.m_inGrad.copy();
		m_training = module.m_training;
	}
	
	BatchNorm &operator=(const BatchNorm &module)
	{
		m_means			= module.m_means.copy();
		m_invStds		= module.m_invStds.copy();
		m_runningMeans	= module.m_runningMeans.copy();
		m_runningVars	= module.m_runningVars.copy();
		m_normalized	= module.m_normalized.copy();
		m_weights		= module.m_weights.copy();
		m_biases		= module.m_biases.copy();
		m_weightsGrad	= module.m_weightsGrad.copy();
		m_biasesGrad	= module.m_biasesGrad.copy();
		m_momentum		= module.m_momentum;
		m_output		= module.m_output.copy();
		m_inGrad		= module.m_inGrad.copy();
		m_outputShape	= module.m_outputShape;
		m_inputShape	= module.m_inputShape;
		m_training		= module.m_training;
		return *this;
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
	Tensor<T> &weights()
	{
		return m_weights;
	}
	
	/// Get the bias of this module.
	Tensor<T> &bias()
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
	
	// MARK: Serialization
	
	/// Save to a serialized node.
	virtual void save(Serialized &node) const override
	{
		node.set("shape", m_inputShape);
		node.set("runningMeans", m_runningMeans);
		node.set("runningVars", m_runningVars);
		node.set("weights", m_weights);
		node.set("biases", m_biases);
		node.set("training", m_training);
		node.set("momentum", m_momentum);
	}
	
	/// Load from a serialized node.
	virtual void load(const Serialized &node) override
	{
		resizeInputs(node.get<Storage<size_t>>("shape"));
		node.get("runningMeans", m_runningMeans);
		node.get("runningVars", m_runningVars);
		node.get("weights", m_weights);
		node.get("biases", m_biases);
		node.get("training", m_training);
		node.get("momentum", m_momentum);
	}
	
	// MARK: Computation
	
	/// Forward propagate input, returning output.
	virtual const Tensor<T> &forward(const Tensor<T> &input) override
	{
		NNAssertEquals(input.shape(), m_inGrad.shape(), "Incompatible input!");
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
	virtual const Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
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
	
	// MARK: Size Management
	
	/// Set the output shape of this module.
	/// In batchnorm, input shape is always equal to output shape.
	virtual void resizeOutputs(const Storage<size_t> &dims) override
	{
		NNAssertEquals(dims.size(), 1, "Expected one-dimensional input!");
		
		Module<T>::inputs(dims);
		Module<T>::outputs(dims);
		
		m_means.resize(dims[0]);
		m_invStds.resize(dims[0]);
		m_runningMeans.resize(dims[0]);
		m_runningVars.resize(dims[0]);
		m_normalized.resizeDim(1, dims);
		m_weights.resize(dims[0]).fill(1);
		m_biases.resize(dims[0]).fill(0);
		m_weightsGrad.resize(dims[0]);
		m_biasesGrad.resize(dims[0]);
	}
	
	/// Set the input shape of this module.
	/// In batchnorm, input shape is always equal to output shape.
	virtual void resizeInputs(const Storage<size_t> &dims) override
	{
		resizeOutputs(dims);
	}
	
	/// Set the input and output shapes of this module.
	/// In batchnorm, input shape is always equal to output shape.
	virtual void resize(const Storage<size_t> &inps, const Storage<size_t> &outs) override
	{
		NNAssertEquals(inps, outs, "Expected input and output sizes to be equal!");
		resizeOutputs(outs);
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
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
	using Module<T>::m_outputShape;
	using Module<T>::m_inputShape;
	using Module<T>::m_training;
	
private:
	Tensor<T> m_means;			///< Mean of each input dimension within the batch.
	Tensor<T> m_invStds;		///< Inverted standard deviation of each input dimension within the batch.
	Tensor<T> m_runningMeans;	///< A running mean for each input, for after training.
	Tensor<T> m_runningVars;	///< A running variance for each input, for after training.
	Tensor<T> m_normalized;		///< Halfway transformed input, cached for backward.
	Tensor<T> m_weights;		///< How much to scale the normalized input.
	Tensor<T> m_biases;			///< How much to shift the normalized input.
	Tensor<T> m_weightsGrad;	///< Gradient of error w.r.t. weights.
	Tensor<T> m_biasesGrad;		///< Gradient of error w.r.t. biases.
	T m_momentum;				///< How much to update running mean and variance.
};

}

NNRegisterType(BatchNorm, Module);

#endif
