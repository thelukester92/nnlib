#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "../module.h"
#include "../util/random.h"

namespace nnlib
{

template <typename T = double>
class Convolution : public Module<T>
{
public:
	struct Shape
	{
		Shape(size_t width, size_t height, size_t channels) : width(width), height(height), channels(channels) {}
		size_t size()
		{
			return width * height * channels;
		}
		size_t width, height, channels;
	};
	
	Convolution(size_t inRows, size_t inCols, size_t inChannels, size_t kernelRows, size_t kernelCols, size_t kernelCount, size_t padding = 0, size_t stride = 1, size_t batch = 1) :
	m_inputShape(inCols, inRows, inChannels),
	m_kernelShape(kernelCols, kernelRows, inChannels),
	m_outputShape((inCols - kernelCols + 2 * padding) / stride + 1, (inRows - kernelRows + 2 * padding) / stride + 1, kernelCount),
	m_padding(padding),
	m_stride(stride),
	m_kernels(kernelCount, m_kernelShape.size() + 1),
	m_kernelsBlame(kernelCount, m_kernelShape.size()),
	m_inputBlame(batch, m_inputShape.size()),
	m_outputs(batch, m_outputShape.size())
	{
		resetWeights();
	}
	
	const Shape &inputShape()
	{
		return m_inputShape;
	}
	
	const Shape &kernelShape()
	{
		return m_kernelShape;
	}
	
	const Shape &outputShape()
	{
		return m_outputShape;
	}
	
	Matrix<T> &kernels()
	{
		return m_kernels;
	}
	
	void resetWeights()
	{
		for(auto &val : m_kernels)
			val = Random<T>::normal(0, 1, 1);
	}
	
	virtual void resize(size_t inps, size_t outs) override
	{
		NNHardAssert(inps == m_inputShape.size() && outs == m_outputShape.size(), "Cannot resize a convolutional module this way!");
		/// \todo allow resizing with special functions
	}
	
	
	virtual Matrix<T> &forward(const Matrix<T> &inputs) override
	{
		NNAssert(inputs.rows() == m_inputBlame.rows(), "Incorrect batch size!");
		NNAssert(inputs.cols() == m_inputBlame.cols(), "Incorrect input size!");
		
		for(size_t inp = 0; inp < inputs.rows(); ++inp)
		{
			for(size_t ker = 0; ker < m_kernels.rows(); ++ker)
			{
				for(size_t outRow = 0; outRow < m_outputShape.height; ++outRow)
				{
					for(size_t outCol = 0; outCol < m_outputShape.width; ++outCol)
					{
						T &sum = m_outputs(inp, (ker * m_outputShape.height + outRow) * m_outputShape.width + outCol);
						sum = m_kernels(ker, m_kernelShape.size());
						for(size_t channel = 0; channel < m_kernelShape.channels; ++channel)
						{
							int inRow = int(outRow * m_stride) - int(m_padding);
							for(size_t row = 0; row < m_kernelShape.height; ++row)
							{
								int inCol = int(outCol * m_stride) - int(m_padding);
								for(size_t col = 0; col < m_kernelShape.width; ++col)
								{
									if(inRow >= 0 && inRow < m_inputShape.height && inCol >= 0 && inCol < m_inputShape.width)
										sum += m_kernels(ker, (channel * m_kernelShape.height + row) * m_kernelShape.width + col) * inputs(inp, (channel * m_inputShape.height + inRow) * m_inputShape.width + inCol);
									++inCol;
								}
								++inRow;
							}
						}
					}
				}
			}
		}
		
		return m_outputs;
	}
	
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		NNAssert(inputs.rows() == m_inputBlame.rows(), "Incorrect batch size!");
		NNAssert(inputs.cols() == m_inputBlame.cols(), "Incorrect input size!");
		NNAssert(blame.rows() == m_outputs.rows(), "Incorrect batch size!");
		NNAssert(blame.cols() == m_outputs.cols(), "Incorrect blame size!");
		
		/// \todo deconvolution
		
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
	
	virtual Vector<Tensor<T> *> parameters() override
	{
		return { &m_kernels };
	}
	
	virtual Vector<Tensor<T> *> blame() override
	{
		return { &m_kernelsBlame };
	}

private:
	Shape m_inputShape;			///< The width, height, and number of channels in the input.
	Shape m_kernelShape;		///< The width, height, and number of channels in the kernels.
	Shape m_outputShape;		///< The width, height, and number of channels in the output.
	size_t m_padding;			///< How many 0s to pad on each edge.
	size_t m_stride;			///< How many pixels to skip between steps.
	Matrix<T> m_kernels;		///< The parameters, imagined as convolutional kernels.
	Matrix<T> m_kernelsBlame;	///< Gradient of the error w.r.t. the parameters.
	Matrix<T> m_inputBlame;		///< Gradient of the error w.r.t. the inputs.
	Matrix<T> m_outputs;		///< The output of this layer.
};

}

#endif // CONVOLUTION_H
