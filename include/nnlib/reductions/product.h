#ifndef PRODUCT_H
#define PRODUCT_H

namespace nnlib
{

/// Product reduction.
template <typename T = double>
class Product
{
public:
	static T init()
	{
		return 1;
	}
	
	static T forward(const T &reduction, const T &x)
	{
		return reduction * x;
	}
	
	static T backward(const T &reduction, const T &x)
	{
		return reduction / x;
	}
};

}

#endif
