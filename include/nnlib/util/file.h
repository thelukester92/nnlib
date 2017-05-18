#ifndef FILE_H
#define FILE_H

#include <string>
#include <string.h>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include <limits>
#include "../tensor.h"

namespace nnlib
{

class Relation
{
public:
	const std::string &name() const
	{
		return m_name;
	}
	
	Relation &name(const std::string &newName)
	{
		m_name = newName;
		return *this;
	}
	
	size_t size() const
	{
		return m_attrNames.size();
	}
	
	size_t size(size_t i) const
	{
		return m_attrVals[i].size();
	}
	
	const std::string &attrName(size_t i)
	{
		return m_attrNames[i];
	}
	
	const std::unordered_map<std::string, size_t> &attrVals(size_t i)
	{
		return m_attrVals[i];
	}
	
	Relation &addAttribute(const std::string &attrName, const std::unordered_map<std::string, size_t> &attrVals)
	{
		m_attrNames.push_back(attrName);
		m_attrVals.push_back(attrVals);
		return *this;
	}
private:
	std::string m_name;
	Storage<std::string> m_attrNames;
	Storage<std::unordered_map<std::string, size_t>> m_attrVals;
};

/// \todo Somehow merge with archive?
template <typename T = double>
class File
{
public:
	static T unknown;
	
	/// Load a weka .arff file.
	static Tensor<T> loadArff(const std::string &filename, Relation *relPtr = nullptr)
	{
		Storage<Tensor<T> *> rows;
		Relation rel;
		
		std::ifstream fin(filename.c_str());
		NNHardAssert(fin.is_open(), "Could not open file '" + filename + "'!");
		
		std::string line;
		while(!fin.fail())
		{
			std::getline(fin, line);
			NNHardAssert(line[0] == '\0' || line[0] == '@' || line[0] == '%', "Invalid arff file!");
			
			if(line[0] == '@')
			{
				if(startsWith(line, "@relation"))
				{
					char *ptr = const_cast<char *>(line.c_str());
					skipWhitespace(&ptr);
					char *end = tokenEnd(ptr);
					rel.name(std::string(ptr, end - ptr));
				}
				else if(startsWith(line, "@attribute"))
				{
					char *ptr = const_cast<char *>(line.c_str() + 10);
					skipWhitespace(&ptr);
					char *end = tokenEnd(ptr);
					std::string attrName(ptr, end - ptr);
					
					ptr = end;
					skipWhitespace(&ptr);
					
					std::unordered_map<std::string, size_t> attrVals;
					if(*ptr == '{')
					{
						++ptr;
						size_t val = 0;
						while(*ptr != '}' && *ptr != '\0')
						{
							skipWhitespace(&ptr);
							char *end = tokenEnd(ptr, ",}");
							attrVals[std::string(ptr, end - ptr)] = val++;
							ptr = end;
							skipWhitespace(&ptr);
							if(*ptr != '\0' && *ptr != '}')
								++ptr;
						}
					}
					else
						NNHardAssert(strncmp(ptr, "numeric", 7) == 0 || strncmp(ptr, "integer", 7) == 0 || strncmp(ptr, "real", 4) == 0, "Unrecognized attribute type!");
					
					rel.addAttribute(attrName, attrVals);
				}
				else if(startsWith(line, "@data"))
					break;
			}
		}
		
		while(!fin.fail())
		{
			std::getline(fin, line);
			char *ptr = const_cast<char *>(line.c_str());
			
			// end-of-file
			if(fin.fail())
				break;
			
			// empty line
			if(*ptr == '\0')
				continue;
			
			Tensor<T> *rowPtr = new Tensor<T>(rel.size());
			Tensor<T> &row = *rowPtr;
			rows.push_back(rowPtr);
			
			size_t i = 0;
			while(*ptr != '\0')
			{
				skipWhitespace(&ptr);
				if(*ptr == '\0')
					break;
				NNHardAssert(i < rel.size(), "Too many columns on row " + std::to_string(rows.size()));
				if(rel.size(i) == 0)
				{
					if(*ptr == '?')
					{
						row(i) = unknown;
						++ptr;
					}
					else
						row(i) = std::strtod(ptr, &ptr);
					
					if(*ptr == ',')
						++ptr;
				}
				else
				{
					char *end = tokenEnd(ptr, ",");
					auto j = rel.attrVals(i).find(std::string(ptr, end - ptr));
					NNHardAssert(j != rel.attrVals(i).end(), "Invalid nominal value '" + std::string(ptr, end - ptr) + "'");
					row(i) = j->second;
					ptr = end;
				}
				++i;
			}
			NNHardAssert(i == rel.size(), "Not enough columns on row " + std::to_string(rows.size()));
		}
		fin.close();
		
		if(relPtr != nullptr)
			*relPtr = rel;
		
		Tensor<T> flattened = Tensor<T>::flatten(rows).resize(rows.size(), rel.size());
		for(auto *i : rows)
			delete i;
		
		for(size_t i = 0; i < flattened.size(1); ++i)
		{
			double sum = 0.0;
			size_t count = 0;
			for(size_t j = 0; j < flattened.size(0); ++j)
			{
				if(flattened(j, i) != unknown)
				{
					sum += flattened(j, i);
					++count;
				}
			}
			if(count > 0)
			{
				double mean = sum / count;
				for(size_t j = 0; j < flattened.size(0); ++j)
					if(flattened(j, i) == unknown)
						flattened(j, i) = mean;
			}
		}
		
		return flattened;
	}
	
	/// Save a weka .arff file.
	static void saveArff(const Tensor<T> &m, const std::string &filename, Relation *relPtr = nullptr)
	{
		NNHardAssert(m.dims() == 2, "Can only save a matrix to an arff file!");
		
		std::ofstream fout(filename.c_str());
		NNHardAssert(fout.is_open(), "Could not open file '" + filename + "'!");
		
		if(relPtr != nullptr)
		{
			NNHardAssert(relPtr->size() == m.size(1), "Incompatible relation!");
			fout << "@relation " << quoted(relPtr->name()) << "\n";
			for(size_t i = 0; i < relPtr->size(); ++i)
			{
				fout << "@attribute " << quoted(relPtr->attrName(i)) << " ";
				if(relPtr->size(i) == 0)
					fout << "real";
				else
				{
					bool first = false;
					fout << "{";
					for(auto &p : relPtr->attrVals(i))
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
			for(size_t i = 0; i < m.size(1); ++i)
				fout << "@attribute attr" << i << " real\n";
		}
		
		fout << "@data\n";
		
		for(size_t i = 0; i < m.size(0); ++i)
		{
			if(m(i, 0) == unknown)
				fout << "?";
			else
				fout << m(i, 0);
			
			for(size_t j = 1; j < m.size(1); ++j)
			{
				fout << ",";
				if(m(i, j) == unknown)
					fout << "?";
				else
					fout << m(i, j);
			}
			
			fout << "\n";
		}
	}
	
private:
	static bool startsWith(std::string str, std::string prefix)
	{
		std::transform(str.begin(), str.end(), str.begin(), ::tolower);
		std::transform(prefix.begin(), prefix.end(), prefix.begin(), ::tolower);
		return str.compare(0, prefix.length(), prefix) == 0;
	}
	
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
			NNHardAssert(*ptr == '\'', "Invalid token!");
			++ptr;
		}
		else if(*ptr == '"')
		{
			++ptr;
			while(*ptr != '"' && *ptr != '\0')
				++ptr;
			NNHardAssert(*ptr == '"', "Invalid token!");
			++ptr;
		}
		else
		{
			while(strspn(ptr, delim) == 0 && *ptr != '\0')
				++ptr;
		}
		
		return ptr;
	}
	
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

template <typename T>
T File<T>::unknown = std::numeric_limits<T>::lowest();

}

#endif
