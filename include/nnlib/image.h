#ifndef IMAGE_H
#define IMAGE_H

#include "matrix.h"

namespace nnlib
{

class Image
{
public:
	/// Takes images (as row vectors) and returns a matrix of patches,
	/// in which each row is a patch or region in the original images.
	/// patches per image = patches.rows / image.rows
	static Matrix patches(
		const Matrix &images,
		size_t imWidth, size_t imHeight, size_t imChannels, bool imInterlaced,
		size_t pWidth, size_t pHeight,
		size_t padX, size_t padY, size_t strideX, size_t strideY
		)
	{
		size_t rowsPerImage = (imHeight - pHeight + 2 * padY) / strideY + 1;
		size_t colsPerChannel = (imWidth - pWidth + 2 * padX) / strideX + 1;
		
		Matrix p(rowsPerImage * images.rows(), colsPerChannel * imChannels);
		
		if(imInterlaced)
		{
			for(size_t i = 0; i < images.rows(); ++i)
				for(size_t row = 0; row < rowsPerImage; ++row)
					for(size_t col = 0; col < colsPerChannel; ++col)
						for(size_t channel = 0; channel < imChannels; ++channel)
							p(i * rowsPerImage + row, colsPerChannel * channel + col) = images(i, (row * imWidth + col) * imChannels + channel);
		}
		else
		{
			for(size_t i = 0; i < images.rows(); ++i)
				for(size_t row = 0; row < rowsPerImage; ++row)
					for(size_t col = 0; col < colsPerChannel; ++col)
						for(size_t channel = 0; channel < imChannels; ++channel)
							p(i * rowsPerImage + row, imChannels * col + channel) = images(i, (channel * imHeight + row) * imWidth + col);
		}
		
		return p;
	}
};

}

#endif
