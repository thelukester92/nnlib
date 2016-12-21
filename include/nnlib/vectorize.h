#ifndef VECTORIZE_H
#define VECTORIZE_H

#include "module.h"

namespace nnlib
{

template <typename T = double>
class Vectorize : public Module<T>
{
public:
	Vectorize(size_t width, size_t height, size_t channels, bool interlaced, size_t kWidth, size_t kHeight, size_t paddingX, size_t paddingY, size_t strideX, size_t strideY, size_t batch = 1)
	: m_width(width), m_height(height), m_channels(channels), m_interlaced(interlaced)
	  m_kWidth(kWidth), m_kHeight(kHeight),
	  m_paddingX(paddingX), m_paddingY(paddingY),
	  m_strideX(strideX), m_strideY(strideY),
	  m_outputs(
	    ((width - kWidth + 2 * paddingX) / strideX + 1) * ((height - kHeight + 2 * paddingY) / strideY + 1),
	    channels * kWidth * kHeight
	  )
	{}
	
	virtual Matrix<T> &forward(const Matrix<T> &inputs) override
	{
		size_t i = 0;
		for(int y = -m_paddingY, y + m_kHeight < m_height + m_paddingY; ++y)
		{
			for(int x = -m_paddingX; x + m_kWidth < m_width + m_paddingX; ++x)
			{
				Vector &row = m_outputs[i++];
				size_t j = 0;
				for(int z = 0; z < m_channels; ++z)
					for(int ky = 0; ky < m_kHeight; ++ky)
						for(int kx = 0; kx < m_kWidth; ++kx)
							row[j++] = at(inputs, i, z, ky + y, kx + x);
			}
		}
		return m_outputs;
	}

private:
	const T &at(const Matrix<T> &inputs, size_t i, int z, int y, int x)
	{
		int idx = index(z, y, x);
		return idx >= 0 ? inputs(i, idx) : 0;
	}
	
	int index(int z, int y, int x)
	{
		if(y >= 0 && y < m_height && x >= 0 && x < m_width)
		{
			if(m_interlaced)
				return ((y * m_width) + x) * m_channels + z;
			else
				return ((z * m_height) + y) * m_width + x;
		}
		else
			return -1;
	}
	
	/// Input image dimensions.
	size_t m_width, m_height, m_channels;
	bool m_interlaced;
	
	/// Window size.
	size_t m_kWidth, m_kHeight;
	
	/// Viewport information.
	size_t m_paddingX, m_paddingY, m_strideX, m_strideY;
	
	/// Buffers.
	Matrix<T> m_outputs;
};

}

#endif
