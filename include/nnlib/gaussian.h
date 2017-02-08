#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include "module.h"
#include "random.h"
#include <limits>

namespace nnlib
{

/// An activation function layer that applies a Gaussian function to each input.
/// Has an adaptive mean and stddev per input.
template <typename T = double>
class Gaussian : public Module<T>
{
public:
	Gaussian(size_t size = 0, size_t batch = 1)
	: m_means(size), m_stddevs(size),
	  m_meanBlame(size), m_stddevBlame(size),
	  m_inputBlame(batch, size), m_outputs(batch, size)
	{
		resetWeights();
	}
	
	void resetWeights()
	{
		for(auto &val : m_means)
			val = Random<T>::uniform(std::numeric_limits<T>::epsilon(), 1.0);
		for(auto &val : m_stddevs)
			val = Random<T>::uniform(1.0);
	}
	
	virtual void resize(size_t size) override
	{
		Module<T>::resize(size, size, m_outputs.rows());
		m_means.resize(size);
		m_stddevs.resize(size);
		m_meanBlame.resize(size);
		m_stddevBlame.resize(size);
		resetWeights();
	}
	
	virtual Matrix<T> &forward(const Matrix<T> &inputs) override
	{
		NNAssert(inputs.rows() == m_inputBlame.rows(), "Incorrect batch size!");
		NNAssert(inputs.cols() == m_inputBlame.cols(), "Incorrect input size!");
		auto i = inputs.begin();
		auto j = m_outputs.begin();
		auto m = m_means.begin(), s = m_stddevs.begin(), end = m_means.end();
		
		for(size_t row = 0; row < inputs.rows(); ++row)
			for(m = m_means.begin(), s = m_stddevs.begin(); m != end; ++m, ++s)
				*j = exp(-(*i - *m) * (*i - *m) / (2 * *s * *s));
		
		return m_outputs;
	}
	
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		NNAssert(inputs.rows() == m_inputBlame.rows(), "Incorrect batch size!");
		NNAssert(inputs.cols() == m_inputBlame.cols(), "Incorrect input size!");
		NNAssert(blame.rows() == m_outputs.rows(), "Incorrect batch size!");
		NNAssert(blame.cols() == m_outputs.cols(), "Incorrect blame size!");
		
		auto i = m_outputs.begin(), j = m_inputBlame.begin();
		auto k = blame.begin();
		auto inp = inputs.begin();
		
		for(size_t row = 0; row < inputs.rows(); ++row)
		{
			auto mean = m_means.begin(), meanBlame = m_meanBlame.begin(), end = m_means.end();
			auto stddev = m_stddevs.begin(), stddevBlame = m_stddevBlame.begin();
			
			m_meanBlame.fill(0);
			m_stddevBlame.fill(0);
			
			for(; mean != end; ++mean, ++meanBlame, ++stddev, ++stddevBlame, ++i, ++j, ++k, ++inp)
			{
				*j = *k * *i * -(*inp - *mean) * (*inp - *mean) / (*stddev * *stddev);
				*meanBlame += *k * *i * -(*inp - *mean) * (*inp - *mean) / (*stddev * *stddev);
				*stddevBlame += *k * *i * (*inp - *mean) * (*inp - *mean) / (*stddev * *stddev * *stddev);
			}
		}
		
		return m_inputBlame;
	}
	
	virtual Matrix<T> &output() override
	{
		return m_outputs;
	}
	
	virtual Matrix<T> &inputBlame() override
	{
		return m_inputBlame;
	}
	
private:
	Vector<T> m_means;			///< Means for each Gaussian function.
	Vector<T> m_stddevs;		///< Standard deviations for each Gaussian function.
	Vector<T> m_meanBlame;		///< Gradient of the error w.r.t. the means.
	Vector<T> m_stddevBlame;	///< Gradient of the error w.r.t. the standard deviations.
	Matrix<T> m_inputBlame;		///< Gradient of the error w.r.t. the inputs.
	Matrix<T> m_outputs;		///< The output of this layer.
};

}

#endif
