#ifndef JSON_SERIALIZER_H
#define JSON_SERIALIZER_H

#include <fstream>
#include <iostream>
#include <limits>

#include "parser.h"
#include "serialized.h"

namespace nnlib
{

class JSONSerializer
{
public:
	JSONSerializer() = delete;
	JSONSerializer(const JSONSerializer &) = delete;
	JSONSerializer &operator=(const JSONSerializer &) = delete;
	
	static Serialized read(std::istream &in)
	{
		Serialized root;
		return root;
	}
	
	static Serialized read(const std::string &filename)
	{
		ifstream fin(filename);
		Serialized result = read(fin);
		fin.close();
		return result;
	}
	
	static void write(const Serialized &root, std::ostream &out, bool pretty = false)
	{
		out.precision(std::numeric_limits<double>::digits10);
		write(root, out, pretty ? 0 : -1);
	}
	
	static void write(const Serialized &root, const std::string &filename, bool pretty = false)
	{
		ofstream fout(filename);
		write(root, fout, pretty);
		fout.close();
	}
	
private:
	static void indent(std::ostream &out, int level)
	{
		for(int i = 0; i < level; ++i)
			out << "\t";
	}
	
	static void newline(std::ostream &out, int level)
	{
		if(level >= 0)
			out << "\n";
	}
	
	static void write(const Serialized &root, std::ostream &out, int level)
	{
		switch(root.type())
		{
		case Serialized::Null:
			out << "null";
			break;
		case Serialized::Boolean:
			out << (root.as<bool>() ? "true" : "false");
			break;
		case Serialized::Integer:
			out << root.as<int>();
			break;
		case Serialized::Float:
			out << root.as<double>();
			break;
		case Serialized::String:
			write(root.as<std::string>(), out);
			break;
		case Serialized::Array:
			write(root.as<SerializedArray>(), out, level);
			break;
		case Serialized::Object:
			write(root.as<SerializedObject>(), out, level);
			break;
		}
	}
	
	static void write(const std::string &str, std::ostream &out)
	{
		out << '"';
		for(char c : str)
		{
			if(c == '"' || c == '\\')
				out << "\\";
			out << c;
		}
		out << '"';
	}
	
	static void write(const SerializedArray &arr, std::ostream &out, int level)
	{
		if(arr.size() == 0)
		{
			out << "[]";
			return;
		}
		
		out << '[';
		
		size_t i = 0;
		for(Serialized *node : arr)
		{
			if(i > 0)
				out << ',';
			
			newline(out, level);
			indent(out, level >= 0 ? level + 1 : level);
			write(*node, out, level >= 0 ? level + 1 : level);
			
			++i;
		}
		
		newline(out, level);
		indent(out, level);
		out << ']';
	}
	
	static void write(const SerializedObject &obj, std::ostream &out, int level)
	{
		if(obj.size() == 0)
		{
			out << "{}";
			return;
		}
		
		out << '{';
		
		size_t i = 0;
		for(const std::string &key : obj)
		{
			if(i > 0)
				out << ',';
			
			newline(out, level);
			indent(out, level >= 0 ? level + 1 : level);
			write(key, out, level >= 0 ? level + 1 : level);
			out << ": ";
			write(*obj[key], out, level >= 0 ? level + 1 : level);
			
			++i;
		}
		
		newline(out, level);
		indent(out, level);
		out << '}';
	}
};

}

#endif
