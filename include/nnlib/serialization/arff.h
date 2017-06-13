#ifndef SERIALIZATION_ARFF_H
#define SERIALIZATION_ARFF_H

#include "csv.h"
#include <iostream>

namespace nnlib
{

/// \brief Serialize to and from ARFF streams.
///
/// Because this serializer is limited in what it can do...
///   1) It only reads/writes matrices
///   2) It only reads/writes a single thing
/// ...this is not an Archive, but a utility class.
class ARFFUtil
{
public:
	ARFFUtil() = delete;
	
	template <typename T = double>
	static void read(Tensor<T> &matrix, std::istream &in)
	{
		Storage<Tensor<T> *> rows;
		std::string line, token;
		
		std::string relationName = "";
		Storage<std::string> attributeNames;
		Storage<std::string> attributeTypes;
		
		// read header
		while(true)
		{
			in >> token;
			toLower(token);
			
			std::cout << "token is :" << token << ":" << std::endl;
			
			NNHardAssert(
				token == "%" || token == "@relation" || token == "@attribute" || token == "@data",
				"Unexpected token '" + token + "'!"
			);
			
			if(token[0] == '%')
			{
				// ignore comments
				std::getline(in, token);
			}
			else if(token == "@relation")
			{
				NNHardAssert(relationName == "", "Cannot assign relation name more than once!");
				readString(in, relationName);
				std::cout << "read relation name: '" << relationName << "'" << std::endl;
				std::getline(in, token);
				NNAssertEquals(token, "", "Unexpected token after relation name!");
			}
			else if(token == "@attribute")
			{
				NNHardAssert(relationName != "", "Cannot assign attributes before assigning relation name!");
				
				attributeNames.push_back("");
				readString(in, attributeNames.back());
				std::cout << "read attribute name: '" << attributeNames.back() << "'" << std::endl;
				
				attributeTypes.push_back("");
				readString(in, attributeTypes.back(), false);
				std::cout << "read attribute type: '" << attributeTypes.back() << "'" << std::endl;
			}
			else if(token == "@data")
			{
				NNHardAssert(
					relationName != "" && attributeNames.size() > 0,
					"Cannot print data before assigning relation name and attributes!"
				);
				break;
			}
		}
		
		
		
		/*
		Tensor<T> row;
		while(readRow(in, row, sep))
		{
			// skip blank lines
			if(row.size(0) == 0)
				continue;
			
			NNAssert(attributes == 0 || row.size(0) == attributes, "Unexpected row size!");
			rows.push_back(new Tensor<T>(row.copy()));
			
			// if not specified, use the number of columns in the first row as attributes
			if(attributes == 0)
				attributes = row.size(0);
		}
		
		matrix = Tensor<T>::flatten(rows).resize(rows.size(), row.size(0));
		for(Tensor<T> *row : rows)
			delete row;
		*/
	}
	
	template <typename T = double>
	static void write(const Tensor<T> &matrix, std::ostream &out, char sep = ',')
	{
		/*
		NNAssertEquals(matrix.dims(), 2, "Only matrices may be written as CSV!");
		
		for(size_t i = 0, rows = matrix.size(0); i != rows; ++i)
		{
			for(size_t j = 0, cols = matrix.size(1); j != cols; ++j)
			{
				if(j > 0)
					out << sep;
				out << matrix(i, j);
			}
			out << std::endl;
		}
		*/
	}
	
private:
	static void toLower(std::string &s)
	{
		for(char &c : s)
			c = tolower(c);
	}
	
	static bool contains(const std::string &seps, char c)
	{
		for(const char &sep : seps)
			if(sep == c)
				return true;
		return false;
	}
	
	static bool readString(std::istream &in, std::string &str, bool canQuote = true, std::string seps = " \t\n")
	{
		char c;
		if(!(in >> c))
			return false;
		
		bool quoted = false, escaped = false;
		if(canQuote && contains("\"'", c))
		{
			quoted = true;
			seps = c;
		}
		else
			in.unget();
		
		str = "";
		while(true)
		{
			if(!in.get(c))
				return false;
			else if(escaped)
				escaped = false;
			else if(c == '\\')
				escaped = true;
			else if(contains(seps, c))
				break;
			if(!escaped)
				str += c;
		}
		
		return true;
	}
	
	template <typename T>
	static bool readRow(std::istream &in, Tensor<T> &row, char sep)
	{
		std::string line, value;
		
		if(!std::getline(in, line))
			return false;
		
		if(line == "")
		{
			row.resize(0);
			return true;
		}
		
		std::istringstream ss(line);
		Storage<T> &storage = row.storage().resize(0);
		while(std::getline(ss, value, sep))
			storage.push_back(std::stod(value));
		row.resize(storage.size());
		
		return true;
	}
};

}

#endif
