#ifndef ARCHIVE_H
#define ARCHIVE_H

#include "serialized.h"
#include <iostream>

namespace nnlib
{

class Archive
{
public:
	static void output(const Serialized &node, std::ostream &out = std::cout)
	{
		switch(node.type())
		{
		case Serialized::Null:
			out << "null";
			break;
		case Serialized::Boolean:
			out << (node.as<bool>() ? "true" : "false");
			break;
		case Serialized::Integer:
			out << node.as<int>();
			break;
		case Serialized::Float:
			out << node.as<double>();
			break;
		case Serialized::String:
			out << "\"" << node.as<std::string>() << "\"";
			break;
		case Serialized::Array:
			output(node.as<SerializedArray>(), out);
			break;
		case Serialized::Object:
			output(node.as<SerializedObject>(), out);
			break;
		}
	}
	
private:
	static void output(const SerializedArray &array, std::ostream &out)
	{
		out << "[ ";
		
		size_t i = 0;
		for(Serialized *node : array)
		{
			if(i > 0)
				out << ", ";
			output(*node, out);
			++i;
		}
		
		out << " ]";
	}
	
	static void output(const SerializedObject &object, std::ostream &out)
	{
		out << "{ ";
		
		size_t i = 0;
		for(const std::string &key : object)
		{
			if(i > 0)
				out << ", ";
			out << "\"" << key << "\": ";
			output(*object[key], out);
			++i;
		}
		
		out << " }";
	}
};

}

#endif
