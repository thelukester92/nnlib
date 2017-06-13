#ifndef SERIALIZATION_CSV_H
#define SERIALIZATION_CSV_H

#include "../error.h"
#include <iostream>

namespace nnlib
{

/// \brief Serialize to and from CSV streams.
///
/// Because this serializer is limited in what it can do...
///   1) It only reads/writes matrices
///   2) It only reads/writes a single thing
/// ...this is not an Archive, but a utility class.
class CSVUtil
{
public:
	CSVUtil() = delete;
	
	template <typename T = double>
	static void read(Tensor<T> &matrix, std::istream &in, char sep = ',', size_t attributes = 0)
	{
		Storage<Tensor<T> *> rows;
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
	}
	
	template <typename T = double>
	static void write(const Tensor<T> &matrix, std::ostream &out, char sep = ',')
	{
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
	}
	
private:
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
