#ifndef SAVER_H
#define SAVER_H

#include "loader.h"

namespace nnlib
{

template <typename T = double>
class Saver
{
using Relation = typename Loader<T>::Relation;
public:
	/// Save a weka .arff file.
	static void saveArff(const Matrix<T> &m, const std::string &filename, Relation *relPtr = nullptr)
	{
		std::ofstream fout(filename.c_str());
		NNHardAssert(fout.is_open(), "Could not open file '" + filename + "'!");
		
		if(relPtr != nullptr)
		{
			NNHardAssert(relPtr->attrNames.size() == m.cols(), "Incompatible relation!");
			fout << "@relation " << quoted(relPtr->name) << "\n";
			for(size_t i = 0; i < relPtr->attrNames.size(); ++i)
			{
				fout << "@attribute " << quoted(relPtr->attrNames[i]) << " ";
				if(relPtr->attrVals[i].size() == 0)
					fout << "real";
				else
				{
					bool first = false;
					fout << "{";
					for(auto &p : relPtr->attrVals[i])
					{
						if(first)
							first = false;
						else
							fout << ",";
						fout << quoted(p.first);
					}
					fout << "}";
				}
			}
		}
		else
		{
			fout << "@relation Untitled\n";
			for(size_t i = 0; i < m.cols(); ++i)
				fout << "@attribute attr" << i << " real\n";
		}
		
		fout << "@data\n";
		
		for(size_t i = 0; i < m.rows(); ++i)
		{
			fout << m(i, 0);
			for(size_t j = 1; j < m.cols(); ++j)
				fout << "," << m(i, j);
			fout << "\n";
		}
	}

private:
	static std::string quoted(const std::string &str)
	{
		if(str.find_first_of(" ") != std::string::npos)
		{
			std::string s = str;
			size_t f, offset = 0;
			while((f = s.find_first_of("\"", offset)) != std::string::npos)
			{
				s.replace(f, 1, "\\\"");
				offset = f + 2;
			}
			return "\"" + s + "\"";
		}
		else
			return str;
	}
};

}

#endif
