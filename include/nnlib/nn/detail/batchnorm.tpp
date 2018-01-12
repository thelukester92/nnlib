#ifndef NN_BATCHNORM_TPP
#define NN_BATCHNORM_TPP

#include "../batchnorm.hpp"

namespace nnlib
{

template <typename T>
BatchNorm<T>::BatchNorm(size_t inps) :
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

template <typename T>
BatchNorm<T>::BatchNorm(const BatchNorm<T> &module) :
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

template <typename T>
BatchNorm<T>::BatchNorm(const Serialized &node) :
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

template <typename T>
BatchNorm<T> &BatchNorm<T>::operator=(BatchNorm<T> module)
{
	swap(*this, module);
	return *this;
}

template <typename T>
void swap(BatchNorm<T> &a, BatchNorm<T> &b)
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

template <typename T>
BatchNorm<T> &BatchNorm<T>::reset()
{
	if(m_weights.size(0) > 0)
	{
		m_weights.rand();
		m_biases.zeros();
	}
	return *this;
}

template <typename T>
Tensor<T> BatchNorm<T>::weights()
{
	return m_weights;
}

template <typename T>
Tensor<T> BatchNorm<T>::bias()
{
	return m_biases;
}

template <typename T>
T BatchNorm<T>::momentum() const
{
	return m_momentum;
}

template <typename T>
BatchNorm<T> &BatchNorm<T>::momentum(T momentum)
{
	m_momentum = momentum;
	return *this;
}

template <typename T>
bool BatchNorm<T>::isTraining() const
{
	return m_training;
}

template <typename T>
void BatchNorm<T>::training(bool training)
{
	m_training = training;
}

template <typename T>
void BatchNorm<T>::save(Serialized &node) const
{
	node.set("weights", m_weights);
	node.set("biases", m_biases);
	node.set("runningMeans", m_runningMeans);
	node.set("runningVars", m_runningVars);
	node.set("momentum", m_momentum);
	node.set("training", m_training);
}

// MARK: Computation

template <typename T>
Tensor<T> &BatchNorm<T>::forward(const Tensor<T> &input)
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
		{
			forEach([&](T x, T mean, T &y)
			{
				y += (x - mean) * (x - mean);
			}, input.select(0, i), m_means, m_invStds);
		}

		// Update running mean
		forEach([&](T mean, T &runningMean)
		{
			runningMean *= 1 - m_momentum;
			runningMean += m_momentum * mean;
		}, m_means, m_runningMeans);

		// Update running variance (normalize as sample)
		forEach([&](T invStd, T &runningVar)
		{
			runningVar *= 1 - m_momentum;
			runningVar += m_momentum / (n - 1) * invStd;
		}, m_invStds, m_runningVars);

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
		forEach([&](T mean, T invStd, T &y)
		{
			y -= mean;
			y *= invStd;
		}, means, invStds, out);

		// Rescale and reshift using the parameters
		forEach([&](T w, T b, T &y)
		{
			y *= w;
			y += b;
		}, m_weights, m_biases, out);
	}

	return m_output;
}

template <typename T>
Tensor<T> &BatchNorm<T>::backward(const Tensor<T> &input, const Tensor<T> &outGrad)
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

template <typename T>
Storage<Tensor<T> *> BatchNorm<T>::paramsList()
{
	return { &m_weights, &m_biases };
}

template <typename T>
Storage<Tensor<T> *> BatchNorm<T>::gradList()
{
	return { &m_weightsGrad, &m_biasesGrad };
}

template <typename T>
Storage<Tensor<T> *> BatchNorm<T>::stateList()
{
	return Module<T>::stateList().append({ &m_means, &m_invStds, &m_runningMeans, &m_runningVars });
}

}

#endif
