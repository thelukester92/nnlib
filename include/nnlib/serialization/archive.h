#ifndef ARCHIVE_H
#define ARCHIVE_H

#include "serialized_node.h"
#include <iostream>

namespace nnlib
{

class Archive
{
public:
	static void output(const SerializedNode &node, std::ostream &out = std::cout)
	{
		switch(node.type())
		{
		case SerializedNode::Type::Null:
			out << "null";
			break;
		case SerializedNode::Type::Number:
			out << node.as<double>();
			break;
		case SerializedNode::Type::String:
			out << "\"" << node.as<std::string>() << "\"";
			break;
		case SerializedNode::Type::Array:
			output(node.as<SerializedNode::Array>(), out);
			break;
		case SerializedNode::Type::Object:
			output(node.as<SerializedNode::Object>(), out);
			break;
		}
	}
	
private:
	static void output(const SerializedNode::Array &array, std::ostream &out)
	{
		out << "[ ";
		
		size_t i = 0;
		for(SerializedNode *node : array)
		{
			if(i > 0)
				out << ", ";
			output(*node, out);
			++i;
		}
		
		out << " ]";
	}
	
	static void output(const SerializedNode::Object &object, std::ostream &out)
	{
		out << "{ ";
		
		size_t i = 0;
		for(auto &it : object)
		{
			if(i > 0)
				out << ", ";
			out << "\"" << it.first << "\": ";
			output(*it.second, out);
			++i;
		}
		
		out << " }";
	}
};

}

#endif
