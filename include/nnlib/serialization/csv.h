#ifndef SERIALIZATION_CSV_H
#define SERIALIZATION_CSV_H

#include "archive.h"
#include <algorithm>
#include <string>
#include <limits>

namespace nnlib
{

/// Util functions for CSVInputArchive and CSVOutputArchive.
struct CSVArchive
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
	
	CSVArchive() = delete;
	
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
	
	template <typename T>
	static T unknown()
	{
		return std::numeric_limits<T>::lowest();
	}
};

/// Read in a matrix from a CSV file.
class CSVInputArchive : public InputArchive<CSVInputArchive>
{
public:
	CSVInputArchive(std::istream &in, bool arff = false, CSVArchive::Relation *rel = nullptr) :
		InputArchive<CSVInputArchive>(this),
		m_in(in),
		m_arff(arff),
		m_rel(rel)
	{
		NNAssertEquals(m_arff, rel != nullptr, "rel cannot be used unless arff is true!");
	}
	
	template <typename T>
	void process(Tensor<T> &arg)
	{
		Storage<Tensor<T> *> rows;
		CSVArchive::Relation rel;
		std::string line;
		
		// read in the arff header
		if(m_arff)
		{
			do
			{
				std::getline(m_in, line);
				NNHardAssert(line[0] == '\0' || line[0] == '@' || line[0] == '%', "Invalid arff file!");
				
				if(CSVArchive::startsWith(line, "@relation"))
				{
					char *ptr = const_cast<char *>(line.c_str());
					CSVArchive::skipWhitespace(&ptr);
					char *end = CSVArchive::tokenEnd(ptr);
					rel.name(std::string(ptr, end - ptr));
				}
				else if(CSVArchive::startsWith(line, "@attribute"))
				{
					char *ptr = const_cast<char *>(line.c_str() + 10);
					CSVArchive::skipWhitespace(&ptr);
					char *end = CSVArchive::tokenEnd(ptr);
					std::string attrName(ptr, end - ptr);
					
					ptr = end;
					CSVArchive::skipWhitespace(&ptr);
					
					std::unordered_map<std::string, size_t> attrVals;
					if(*ptr == '{')
					{
						++ptr;
						size_t val = 0;
						while(*ptr != '}' && *ptr != '\0')
						{
							CSVArchive::skipWhitespace(&ptr);
							char *end = CSVArchive::tokenEnd(ptr, ",}");
							attrVals[std::string(ptr, end - ptr)] = val++;
							ptr = end;
							CSVArchive::skipWhitespace(&ptr);
							if(*ptr != '\0' && *ptr != '}')
								++ptr;
						}
					}
					else
						NNHardAssert(strncmp(ptr, "numeric", 7) == 0 || strncmp(ptr, "integer", 7) == 0 || strncmp(ptr, "real", 4) == 0, "Unrecognized attribute type!");
					
					rel.addAttribute(attrName, attrVals);
				}
			}
			while(!m_in.fail() && !CSVArchive::startsWith(line, "@data"));
		}
		
		while(!m_in.eof())
		{
			std::getline(m_in, line);
			char *ptr = const_cast<char *>(line.c_str());
			
			// end-of-file
			if(m_in.fail())
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
				CSVArchive::skipWhitespace(&ptr);
				if(*ptr == '\0')
					break;
				NNHardAssert(i < rel.size(), "Too many columns on row " + std::to_string(rows.size()));
				if(rel.size(i) == 0)
				{
					if(*ptr == '?')
					{
						row(i) = CSVArchive::unknown<T>();
						++ptr;
					}
					else
						row(i) = std::strtod(ptr, &ptr);
					
					if(*ptr == ',')
						++ptr;
				}
				else
				{
					char *end = CSVArchive::tokenEnd(ptr, ",");
					auto j = rel.attrVals(i).find(std::string(ptr, end - ptr));
					NNHardAssert(j != rel.attrVals(i).end(), "Invalid nominal value '" + std::string(ptr, end - ptr) + "'");
					row(i) = j->second;
					ptr = end;
				}
				++i;
			}
			NNHardAssert(i == rel.size(), "Not enough columns on row " + std::to_string(rows.size()));
		}
		
		if(m_rel != nullptr)
			*m_rel = rel;
		
		Tensor<T> flattened = Tensor<T>::flatten(rows).resize(rows.size(), rel.size());
		for(auto *i : rows)
			delete i;
		
		for(size_t i = 0; i < flattened.size(1); ++i)
		{
			double sum = 0.0;
			size_t count = 0;
			for(size_t j = 0; j < flattened.size(0); ++j)
			{
				if(flattened(j, i) != CSVArchive::unknown<T>())
				{
					sum += flattened(j, i);
					++count;
				}
			}
			if(count > 0)
			{
				double mean = sum / count;
				for(size_t j = 0; j < flattened.size(0); ++j)
					if(flattened(j, i) == CSVArchive::unknown<T>())
						flattened(j, i) = mean;
			}
		}
		
		arg = flattened;
	}
	
private:
	std::istream &m_in;
	bool m_arff;
	CSVArchive::Relation *m_rel;
};

/// Write a matrix to a CSV file.
class CSVOutputArchive : public OutputArchive<CSVOutputArchive>
{
public:
	CSVOutputArchive(std::ostream &out) :
		OutputArchive<CSVOutputArchive>(this),
		m_out(out)
	{}
	
	template <typename T>
	void process(const Tensor<T> &arg)
	{
		/// \todo
	}
	
private:
	std::ostream &m_out;
	bool m_arff;
	CSVArchive::Relation *m_rel;
};

}

/// \todo fix CanSerialize in detail.h then uncomment this
// NNRegisterArchive(CSVInputArchive);
// NNRegisterArchive(CSVOutputArchive);

#endif
