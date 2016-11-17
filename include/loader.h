#ifndef LOADER_H
#define LOADER_H

#include "error.h"
#include "tensor.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <list>

namespace nnlib
{

template <typename T>
class Loader
{
public:
	static Matrix<T> load(const char *filename)
	{
		std::string f(filename);
		std::string ext = f.substr(f.find_last_of(".") + 1);
		std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
		if(ext == "raw")
			return loadRaw(filename);
		else if(ext == "arff")
			return loadArff(filename);
		else
			throw std::runtime_error("Unrecognized file format!");
	}
	
	static Matrix<T> loadRaw(const char *filename)
	{
		size_t r, c;
		std::ifstream fin(filename, std::ios::in | std::ios::binary);
		NNAssert(!fin.fail(), "Could not load file!");
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
		bool isData = false;
		size_t cols = 0;
		T val;
		std::string line, piece;
		std::ifstream fin(filename);
		NNAssert(!fin.fail(), "Could not load file!");
		std::list<Vector<T>> vectors;
		std::getline(fin, line);
		while(!fin.fail())
		{
			std::istringstream iss(line);
			if(!isData)
			{
				if(!line.empty())
				{
					iss >> piece;
					std::transform(piece.begin(), piece.end(), piece.begin(), ::tolower);
					if(piece == "@attribute")
						++cols;
					else if(piece == "@data")
						isData = true;
					else if(piece != "@relation" && piece[0] != '%')
						throw std::runtime_error("Invalid arff file!");
				}
			}
			else
			{
				vectors.push_back(Vector<T>(cols));
				size_t i;
				std::getline(iss, piece, ',');
				for(i = 0; i < cols && !iss.fail(); ++i)
				{
					std::istringstream issp(piece);
					issp >> val;
					vectors.back()[i] = val;
					std::getline(iss, piece, ',');
				}
				NNAssert(i == cols, "Incomplete row in arff file!");
				
			}
			std::getline(fin, line);
		}
		
		Matrix<T> m(vectors.size(), cols);
		auto v = vectors.begin();
		for(size_t i = 0; i < vectors.size(); ++i, ++v)
			for(size_t j = 0; j < cols; ++j)
				m(i, j) = v->at(j);
		fin.close();
		return m;
	}
};

}

#endif
