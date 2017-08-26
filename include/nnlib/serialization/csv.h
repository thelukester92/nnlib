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
		SerializedNode rows((SerializedNode::Array()));
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
					out << "\"" << value->as<std::string>() << "\"";
				
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
		
		SerializedNode *node = new SerializedNode(SerializedNode::Array());
		std::string value;
		
		std::istringstream ss(line);
		while(std::getline(ss, value, sep))
		{
			try
			{
				node->append(std::stod(value));
			}
			catch(const std::invalid_argument &)
			{
				node->append(value);
			}
		}
		
		return node;
	}
};

}

#endif
