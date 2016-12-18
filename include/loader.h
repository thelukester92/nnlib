#ifndef LOADER_H
#define LOADER_H

#include <string>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include "matrix.h"
#include "error.h"

namespace nnlib
{

template <typename T>
class Loader
{
public:
	struct Relation
	{
		std::string name;
		std::vector<std::string> attrNames;
		std::vector<std::unordered_map<std::string, size_t>> attrVals;
	};
	
	/// Load a weka .arff file. Assumes real values (for now).
	static Matrix<T> loadArff(const std::string &filename, Relation *relPtr = nullptr)
	{
		Vector<Tensor<T> *> rows;
		Relation rel;
		
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
				if(line.compare(0, 9, "@relation") == 0)
				{
					char *ptr = const_cast<char *>(line.c_str());
					skipWhitespace(&ptr);
					char *end = tokenEnd(ptr);
					rel.name = std::string(ptr, end - ptr);
				}
				else if(line.compare(0, 10, "@attribute") == 0)
				{
					char *ptr = const_cast<char *>(line.c_str() + 10);
					skipWhitespace(&ptr);
					char *end = tokenEnd(ptr);
					rel.attrNames.push_back(std::string(ptr, end - ptr));
					
					ptr = end;
					skipWhitespace(&ptr);
					
					std::unordered_map<std::string, size_t> attrVals;
					if(*ptr == '{')
					{
						++ptr;
						size_t val = 0;
						while(*ptr != '}' && *ptr != '\0')
						{
							char *end = tokenEnd(ptr, ",}");
							attrVals[std::string(ptr, end - ptr)] = val++;
							ptr = end;
							skipWhitespace(&ptr);
							if(*ptr != '\0' && *ptr != '}')
								++ptr;
						}
					}
					else
						NNAssert(strncmp(ptr, "numeric", 7) == 0 || strncmp(ptr, "integer", 7) == 0 || strncmp(ptr, "real", 4) == 0, "Unrecognized attribute type!");
					
					rel.attrVals.push_back(attrVals);
				}
				else if(line.compare(0, 5, "@data") == 0)
					break;
			}
		}
		while(!fin.fail())
		{
			Vector<T> *rowPtr = new Vector<T>(rel.attrNames.size());
			Vector<T> &row = *rowPtr;
			rows.push_back(rowPtr);
			
			std::getline(fin, line);
			char *ptr = const_cast<char *>(line.c_str());
			
			if(fin.fail())
				break;
			
			size_t i = 0;
			while(*ptr != '\0')
			{
				skipWhitespace(&ptr);
				if(*ptr == '\0')
					break;
				NNAssert(i < rel.attrNames.size(), "Too many columns on row " + std::to_string(rows.size()));
				if(rel.attrVals[i].size() == 0)
					row[i] = std::strtod(ptr, &ptr);
				else
				{
					char *end = tokenEnd(ptr, ",");
					auto j = rel.attrVals[i].find(std::string(ptr, end - ptr));
					NNAssert(j != rel.attrVals[i].end(), "Invalid nominal value '" + std::string(ptr, end - ptr) + "'");
					row[i] = j->second;
					ptr = end - 1;
				}
				++ptr;
				++i;
			}
			NNAssert(i == rel.attrNames.size(), "Not enough columns on row " + std::to_string(rows.size()));
		}
		fin.close();
		
		if(relPtr != nullptr)
			relPtr = new Relation(rel);
		
		/// Flatten and return.
		Matrix<T> flattened(Vector<T>(rows), rows.size(), rel.attrNames.size());
		for(auto *i : rows)
			delete i;
		return flattened;
	}

private:
	static void skipWhitespace(char **ptr)
	{
		while(**ptr == ' ' || **ptr == '\t')
			++*ptr;
	}
	
	static char *tokenEnd(char *start, const char *delim = " \t")
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
			while(strspn(ptr, delim) == 0 && *ptr != '\0')
				++ptr;
		}
		
		return ptr;
	}
};

}

#endif
