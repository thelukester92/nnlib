#ifndef CSV_H
#define CSV_H

#include "serialized.h"
#include <iostream>

namespace nnlib
{

/// \brief Serialize to and from CSV streams.
///
/// Reads and writes nodes that are 2-dimensional arrays.
/// Can be true CSV (comma-separated values); may use other separators such as tab.
class CsvSerializer
{
public:
	CsvSerializer() = delete;
	
	static Serialized read(std::istream &in, char sep = ',')
	{
		Serialized rows;
		Serialized *row;
		while((row = readRow(in, sep)))
			rows.add(row);
		return rows;
	}
	
	static void write(const Serialized &rows, std::ostream &out, char sep = ',')
	{
		for(Serialized *row : rows.as<SerializedArray>())
		{
			size_t idx = 0;
			for(Serialized *value : row->as<SerializedArray>())
			{
				if(idx > 0)
					out << sep;
				
				switch(value->type())
				{
				case Serialized::Null:
					out << "null";
					break;
				case Serialized::Integer:
					out << value->as<int>();
					break;
				case Serialized::Float:
					out << value->as<double>();
					break;
				case Serialized::String:
					writeString(out, value->as<std::string>());
					break;
				default:
					throw Error("Invalid type!");
				}
				
				++idx;
			}
			out << std::endl;
		}
	}
	
private:
	static Serialized *readRow(std::istream &in, char sep)
	{
		std::string line;
		do
		{
			if(!std::getline(in, line))
				return nullptr;
		}
		while(line == "");
		
		Serialized *node = new Serialized();
		std::string value;
		
		std::istringstream ss(line);
		while(ss.peek() != EOF)
		{
			if(ss.peek() == '"' || !std::isdigit(ss.peek()))
			{
				readString(ss, value, sep);
				node->add(value);
			}
			else
			{
				std::getline(ss, value, sep);
				try
				{
					node->add(std::stod(value));
				}
				catch(const std::invalid_argument &e)
				{
					throw Error(e.what());
				}
			}
		}
		
		return node;
	}
	
	static void readString(std::istream &in, std::string &out, char sep)
	{
		if(in.peek() != '"')
			std::getline(in, out, sep);
		else
		{
			std::string piece;
			in.get();
			
			do
			{
				std::getline(in, piece, '"');
				out += piece;
				if(in.peek() == '"')
					out += '"';
			}
			while(in.peek() == '"');
			
			NNHardAssert(in.peek() == EOF || in.peek() == sep, "Invalid CSV file!");
			in.get();
		}
	}
	
	static void writeString(std::ostream &out, const std::string &str)
	{
		out << "\"";
		for(size_t i = 0, end = str.length(); i != end; ++i)
		{
			if(str[i] == '"')
				out << "\"\"";
			else
				out << str[i];
		}
		out << "\"";
	}
};

}

#endif
