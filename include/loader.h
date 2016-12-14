#ifndef LOADER_H
#define LOADER_H

#include <string>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include "matrix.h"
#include "error.h"

namespace nnlib
{

template <typename T>
class Loader
{
public:
	/// Load a weka .arff file. Assumes real values (for now).
	static Matrix<T> loadArff(const std::string &filename)
	{
		Vector<Tensor<T> *> rows;
		size_t cols = 0;
		
		std::ifstream fin(filename.c_str());
		NNAssert(fin.is_open(), "Could not open file '" + filename + "'!");
		
		std::string line;
		while(!fin.fail())
		{
			std::getline(fin, line);
			NNAssert(line[0] == '\0' || line[0] == '@' || line[0] == '%', "Invalid arff file!")
			
			std::transform(line.begin(), line.end(), line.begin(), ::tolower);
			if(line[0] == '@')
			{
				if(line.compare(0, 10, "@attribute") == 0)
					++cols;
				else if(line.compare(0, 5, "@data") == 0)
					break;
			}
		}
		while(!fin.fail())
		{
			Vector<T> *rowPtr = new Vector<T>(cols);
			Vector<T> &row = *rowPtr;
			rows.push_back(rowPtr);
			
			std::getline(fin, line);
			char *ptr = const_cast<char *>(line.c_str());
			
			if(fin.fail())
				break;
			
			size_t i = 0;
			while(*ptr != '\0')
			{
				while(*ptr == ' ' || *ptr == '\t')
					++ptr;
				if(*ptr == '\0')
					break;
				NNAssert(i < cols, "Too many columns on row " + std::to_string(rows.size()));
				row[i] = std::strtod(ptr, &ptr);
				if(*ptr == ',')
					++ptr;
				++i;
			}
			NNAssert(i == cols, "Not enough columns on row " + std::to_string(rows.size()));
		}
		fin.close();
		
		/// Flatten and return.
		return Matrix<T>(Vector<T>(rows), rows.size(), cols);
	}
};

}

#endif
