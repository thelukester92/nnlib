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
		Vector<Vector<std::string>> attributes;
		
		std::ifstream fin(filename.c_str());
		NNAssert(fin.is_open(), "Could not open file '" + filename + "'!");
		
		std::string line;
		while(!fin.fail())
		{
			std::getline(fin, line);
			NNAssert(line[0] == '\0' || line[0] == '@' || line[0] == '%', "Invalid arff file!");
			
			std::transform(line.begin(), line.end(), line.begin(), ::tolower);
			if(line[0] == '@')
			{
				if(line.compare(0, 10, "@attribute") == 0)
				{
					char *ptr = const_cast<char *>(line.c_str() + 10);
					
					// Skip attribute name
					ptr = tokenEnd(ptr);
					
					// Check attribute type
					if(strncmp(ptr, "numeric", 7) == 0 || strncmp(ptr, "integer", 7) == 0 || strncmp(ptr, "real", 4) == 0)
						attributes.push_back(Vector<std::string>());
					else if(*ptr == '{')
					{
						Vector<std::string> vals;
						while(*ptr != '}' && *ptr != '\0')
						{
							char *end = tokenEnd(ptr);
							vals.push_back(std::string(ptr, end - ptr));
							ptr = end;
							skipWhitespace(&ptr);
							if(*ptr != '\0' && *ptr != '}')
								++ptr;
							
							std::cout << "found nominal: '" << vals.back() << "'" << std::endl;
						}
						attributes.push_back(vals);
					}
					else
						NNAssert(false, "Unrecognized attribute type!");
				}
				else if(line.compare(0, 5, "@data") == 0)
					break;
			}
		}
		while(!fin.fail())
		{
			Vector<T> *rowPtr = new Vector<T>(attributes.size());
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
				NNAssert(i < attributes.size(), "Too many columns on row " + std::to_string(rows.size()));
				row[i] = std::strtod(ptr, &ptr);
				if(*ptr == ',')
					++ptr;
				++i;
			}
			NNAssert(i == attributes.size(), "Not enough columns on row " + std::to_string(rows.size()));
		}
		fin.close();
		
		/// Flatten and return.
		return Matrix<T>(Vector<T>(rows), rows.size(), attributes.size());
	}

private:
	static void skipWhitespace(char **ptr)
	{
		while(**ptr == ' ' || **ptr == '\t')
			++*ptr;
	}
	
	static char *tokenEnd(char *start)
	{
		char *ptr = start;
		skipWhitespace(&ptr);
		
		// Skip token; may be quoted
		if(*ptr == '\'')
		{
			++ptr;
			while(*ptr != '\'' && *ptr != '\0')
				++ptr;
			NNAssert(*ptr == '\'', "Invalid token!");
			++ptr;
		}
		else if(*ptr == '"')
		{
			++ptr;
			while(*ptr != '"' && *ptr != '\0')
				++ptr;
			NNAssert(*ptr == '"', "Invalid token!");
			++ptr;
		}
		else
		{
			while(*ptr != ' ' && *ptr != '\t' && *ptr != '\0')
				++ptr;
		}
		
		return ptr;
	}
};

}

#endif
