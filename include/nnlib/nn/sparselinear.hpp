#ifndef NN_SPARSE_LINEAR_HPP
#define NN_SPARSE_LINEAR_HPP

#include "linear.hpp"

namespace nnlib
{

/// \brief A feed-forward layer that returns a linear combination of sparse inputs.
///
/// See nnlib/util/tensor_util.hpp for an explanation of sparse tensors.
/// Like a Linear layer, a SparseLinear layer expects either vector or matrix input,
/// so the input must be a matrix that encodes a vector or matrix.
template <typename T = double>
class SparseLinear : public Linear<T>
{
public:
	using Linear<T>::Linear;
	
	// MARK: Computation
	
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		NNAssertEquals(input.dims(), 2, "Sparse tensors must be represented as matrices!");
		
		if(input.size(1) == 2)
		{
			NNAssertEquals(input(0, 0), m_weights.size(0), "Incompatible size indicated by sparse tensor!");
			
			// sparse vector input
			m_output.resize(m_weights.size(1));
			
			// bias
			if(m_bias)
				m_output.copy(*m_bias);
			else
				m_output.zeros();
			
			// sparse matrix/vector multiplication
			for(size_t i = 1, end = input.size(0); i != end; ++i)
				m_output.addV(m_weights.select(0, input(i, 0)), input(i, 1));
		}
		else if(input.size(1) == 3)
		{
			NNAssertEquals(input(0, 1), m_weights.size(0), "Incompatible size indicated by sparse tensor!");
			
			// sparse matrix input
			m_output.resize(input(0, 0), m_weights.size(1)).zeros();
			
			// sparse matrix/matrix multiplication
			for(size_t i = 1, end = input.size(0); i != end; ++i)
				m_output.select(0, input(i, 0)).addV(m_weights.select(0, input(i, 1)), input(i, 2));
			
			// bias
			if(m_bias)
				m_output.assignVV(m_ones.resize(input(0, 0)).fill(1), *m_bias, 1, 1);
		}
		else
		{
			throw Error("Expected sparse vector or sparse matrix input!");
		}
		
		return m_output;
	}
	
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssertEquals(input.dims(), 2, "Sparse tensors must be represented as matrices!");
		
		if(input.size(1) == 2)
		{
			NNAssertEquals(input(0, 0), m_weights.size(0), "Incompatible size indicated by sparse tensor!");
			NNAssertEquals(outGrad.dims(), 1, "Incompatible input and outGrad!");
			NNAssertEquals(outGrad.size(), m_weights.size(1), "Incompatible outGrad!");
			
			// sparse vector outer product
			for(size_t i = 1, end = input.size(0); i != end; ++i)
				m_weightsGrad.select(0, input(i, 0)).addV(outGrad, input(i, 1));
			
			// bias gradient
			if(m_bias)
				m_biasGrad->addV(outGrad);
			
			// input gradient
			m_inGrad.resize(m_weights.size(0));
			m_inGrad.assignMV(m_weights, outGrad);
		}
		else if(input.size(1) == 3)
		{
			NNAssertEquals(input(0, 1), m_weights.size(0), "Incompatible size indicated by sparse tensor!");
			NNAssertEquals(outGrad.dims(), 2, "Incompatible input and outGrad!");
			NNAssertEquals(outGrad.size(0), input(0, 0), "Incompatible outGrad!");
			NNAssertEquals(outGrad.size(1), m_weights.size(1), "Incompatible outGrad!");
			
			// sparse matrix/matrix multiplication
			for(size_t i = 1, end = input.size(0); i != end; ++i)
				m_weightsGrad.select(0, input(i, 1)).addV(outGrad.select(0, input(i, 0)), input(i, 2));
			
			// bias gradient
			if(m_bias)
				m_biasGrad->assignMTV(outGrad, m_ones.resize(input(0, 0)).fill(1), 1, 1);
			
			// input gradient
			m_inGrad.resize(input(0, 0), m_weights.size(0));
			m_inGrad.assignMMT(outGrad, m_weights);
		}
		else
		{
			throw Error("Expected sparse vector or sparse matrix input!");
		}
		
		return m_inGrad;
	}
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
	
	using Linear<T>::m_weights;
	using Linear<T>::m_weightsGrad;
	using Linear<T>::m_bias;
	using Linear<T>::m_biasGrad;
	
	using Linear<T>::m_ones;
};

}

NNRegisterType(SparseLinear, Module);

#endif
