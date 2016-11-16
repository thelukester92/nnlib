#ifndef LOADER_H
#define LOADER_H

#include "error.h"
#include "tensor.h"
#include <fstream>
#include <sstream>

namespace nnlib
{

template <typename T>
class Loader
{
public:
	static Matrix<T> loadRaw(const char *filename)
	{
		size_t r, c;
		std::ifstream fin(filename, std::ios::in | std::ios::binary);
		Assert(!fin.fail(), "Could not load file!");
		fin.read((char *) &r, sizeof(size_t));
		fin.read((char *) &c, sizeof(size_t));
		Matrix<T> m(r, c);
		for(auto i = m.begin(); i != m.end(); ++i)
			fin.read((char *) i, sizeof(T));
		fin.close();
		return m;
	}
	
	/// \todo allow non-continuous attributes
	static Matrix<T> loadArff(const char *filename)
	{
		std::ifstream fin(filename);
		Assert(!fin.fail(), "Could not load file!");
		
		fin.close();
		return m;
	}
};

}

#endif
