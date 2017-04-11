#ifndef SUM_H
#define SUM_H

namespace nnlib
{

/// Sum reduction.
template <typename T = double>
class Sum
{
public:
	static T init()
	{
		return 0;
	}
	
	static T forward(const T &reduction, const T &x)
	{
		return reduction + x;
	}
	
	static T backward(const T &reduction, const T &x)
	{
		return 1;
	}
};

}

#endif
