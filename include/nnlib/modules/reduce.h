#ifndef REDUCE_H
#define REDUCE_H

#include "../module.h"

namespace nnlib
{

/// A module that performs a reduction on the inputs.
/// F must have static forward, backward, and init functions.
/// \todo determine whether F should know its initial value; not standard for reduce.
/// \todo also make a virtual Matrix<T> &forward(const Table<T> &inputs) for flattening
template <template <typename> class F, typename T = double>
class Reduce : public Module<T>
{
public:
	/// A reduction from one size to another.
	Reduce(size_t inps, size_t outs, size_t batch = 1) :
		m_slices(inps / outs),
		m_inputBlame(batch, inps),
		m_output(batch, outs)
	{
		NNAssert(inps % outs == 0, "Invalid number of outputs for the given number of inputs!");
	}
	
	/// A reduction from any size to outs.
	Reduce(size_t outs) :
		m_slices(0),
		m_inputBlame(1, 0),
		m_output(1, outs)
	{}
	
	/// A reduction from any size to 1.
	Reduce() :
		m_slices(1),
		m_inputBlame(1, 0),
		m_output(1, 1)
	{}
	
	virtual void resize(size_t inps, size_t outs) override
	{
		Module<T>::resize(inps, outs);
		NNAssert(inps % outs == 0, "Invalid number of outputs for the given number of inputs!");
		m_slices = inps / outs;
	}
	
	virtual Matrix<T> &forward(const Matrix<T> &inputs) override
	{
		NNAssert(inputs.rows() == this->batchSize() && inputs.cols() == this->inputs(), "Incompatible input!");
		
		Vector<size_t> indices(m_slices);
		
		size_t rows = inputs.rows(), outs = m_output.cols();
		for(size_t row = 0; row < rows; ++row)
		{
			// reset indices
			for(size_t i = 0, index = 0; i < m_slices; ++i, index += outs)
				indices(i) = index;
			
			// reduction
			/// \todo improve performance; can this be parallelized?
			for(size_t i = 0; i < outs; ++i)
			{
				T &reduction = m_output(row, i);
				reduction = F<T>::init();
				for(size_t j = 0; j < m_slices; ++j)
					reduction = F<T>::forward(reduction, inputs(row, indices(j)));
			}
		}
		
		return m_output;
	}
	
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		NNAssert(inputs.rows() == this->batchSize() && inputs.cols() == this->inputs(), "Incompatible input!");
		NNAssert(blame.rows() == this->batchSize() && blame.cols() == this->outputs(), "Incompatible blame!");
		
		Vector<size_t> indices(m_slices);
		
		size_t rows = inputs.rows(), outs = m_output.cols();
		for(size_t row = 0; row < rows; ++row)
		{
			// reset indices
			for(size_t i = 0, index = 0; i < m_slices; ++i, index += outs)
				indices(i) = index;
			
			// reduction
			/// \todo improve performance; can this be parallelized?
			for(size_t i = 0; i < outs; ++i)
			{
				const T &output = m_output(row, i), &blam = blame(row, i);
				for(size_t j = 0; j < m_slices; ++j)
					m_inputBlame(row, indices(j)) = blam * F<T>::backward(output, inputs(row, indices(j)));
			}
		}
		
		return m_inputBlame;
	}
	
	virtual Matrix<T> &output() override
	{
		return m_output;
	}
	
	virtual Matrix<T> &inputBlame() override
	{
		return m_inputBlame;
	}
	
private:
	size_t m_slices;		///< The number of slices (inputs per output). Defaults to
	Matrix<T> m_inputBlame;	///< Gradient of the error w.r.t. the inputs.
	Matrix<T> m_output;		///< The output of this layer.
};

}

#endif
