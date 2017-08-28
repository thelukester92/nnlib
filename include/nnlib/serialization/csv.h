#ifndef CSV_H
#define CSV_H

#include "serialized_node.h"
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
	
	static SerializedNode read(std::istream &in, char sep = ',')
	{
		SerializedNode rows;
		SerializedNode *row;
		while((row = readRow(in, sep)))
			rows.append(row);
		return rows;
	}
	
	static void write(const SerializedNode &rows, std::ostream &out, char sep = ',')
	{
		for(SerializedNode *row : rows.as<SerializedNode::Array>())
		{
			size_t idx = 0;
			for(SerializedNode *value : row->as<SerializedNode::Array>())
			{
				NNHardAssert(value->type() == SerializedNode::Type::Number || value->type() == SerializedNode::Type::String, "Invalid type!");
				
				if(idx > 0)
					out << sep;
				
				if(value->type() == SerializedNode::Type::Number)
					out << value->as<double>();
				else
					writeString(out, value->as<std::string>());
				
				++idx;
			}
			out << std::endl;
		}
	}
	
private:
	static SerializedNode *readRow(std::istream &in, char sep)
	{
		std::string line;
		do
		{
			if(!std::getline(in, line))
				return nullptr;
		}
		while(line == "");
		
		SerializedNode *node = new SerializedNode();
		std::string value;
		
		std::istringstream ss(line);
		while(ss.peek() != EOF)
		{
			if(ss.peek() == '"' || !std::isdigit(ss.peek()))
			{
				readString(ss, value, sep);
				node->append(value);
			}
			else
			{
				std::getline(ss, value, sep);
				try
				{
					node->append(std::stod(value));
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
