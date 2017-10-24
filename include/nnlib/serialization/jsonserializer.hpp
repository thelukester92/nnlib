#ifndef SERIALIZATION_JSON_SERIALIZER_HPP
#define SERIALIZATION_JSON_SERIALIZER_HPP

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

#include "parser.hpp"
#include "serialized.hpp"

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
		Parser p(in);
		Serialized root;
		readValue(root, p);
		return root;
	}
	
	static Serialized readString(const std::string &s)
	{
		std::istringstream iss(s);
		return read(iss);
	}
	
	static Serialized readFile(const std::string &filename)
	{
		std::ifstream fin(filename);
		Serialized result = read(fin);
		fin.close();
		return result;
	}
	
	template <typename T>
	static void write(const T &value, std::ostream &out, bool pretty = false)
	{
		write(Serialized(value), out, pretty);
	}
	
	static void write(const Serialized &root, std::ostream &out, bool pretty = false)
	{
		out.precision(std::numeric_limits<double>::digits10);
		write(root, out, pretty ? 0 : -1);
	}
	
	template <typename T>
	static void writeFile(const T &value, const std::string &filename, bool pretty = false)
	{
		writeFile(Serialized(value), filename, pretty);
	}
	
	static void writeFile(const Serialized &root, const std::string &filename, bool pretty = false)
	{
		std::ofstream fout(filename);
		write(root, fout, pretty);
		fout.close();
	}
	
private:
// MARK: Reading
	
	static void readValue(Serialized &node, Parser &p)
	{
		p.consumeWhitespace();
		NNHardAssert(!p.eof(), "Unexpected end of file!");
		
		char c = p.peek();
		if(c == 'n')
			readNull(node, p);
		else if(c == 't' || c == 'f')
			readBool(node, p);
		else if((c >= '0' && c <= '9') || c == '-')
			readNumber(node, p);
		else if(c == '"')
			readString(node, p);
		else if(c == '[')
			readArray(node, p);
		else if(c == '{')
			readObject(node, p);
	}
	
	static void readNull(Serialized &node, Parser &p)
	{
		NNHardAssert(p.consume("null"), "Expected null!");
		node.type(Serialized::Null);
	}
	
	static void readBool(Serialized &node, Parser &p)
	{
		if(p.peek() == 't')
		{
			NNHardAssert(p.consume("true"), "Expected true!");
			node.set(true);
		}
		else
		{
			NNHardAssert(p.consume("false"), "Expected false!");
			node.set(false);
		}
	}
	
	static void readNumber(Serialized &node, Parser &p)
	{
		std::string intPart, floatPart;
		
		if(p.consume('-'))
			intPart.push_back('-');
		
		if(!p.consume('0'))
			intPart = p.consumeDigits();
		else
			intPart.push_back('0');
		
		if(p.consume('.'))
			floatPart = p.consumeDigits();
		
		if(p.consume('e') || p.consume('E'))
		{
			floatPart.push_back('e');
			
			if(p.consume('-'))
				floatPart.push_back('-');
			else
				p.consume('+');
			
			floatPart += p.consumeDigits();
		}
		
		if(floatPart.length() == 0 && intPart.length() <= 10)
			node.set(std::stoi(intPart));
		else
			node.set(std::stod(intPart + '.' + floatPart));
	}
	
	static void readString(Serialized &node, Parser &p)
	{
		std::string value;
		
		p.ignore();
		while(p.peek() != '"')
		{
			NNHardAssertNotEquals(p.peek(), EOF, "Expected closing quotation mark!");
			
			if(p.consume('\\'))
			{
				NNHardAssertNotEquals(p.peek(), EOF, "Expected escaped character!");
				value.push_back(p.get());
			}
			else
				value.push_back(p.get());
		}
		p.ignore();
		
		node.set(value);
	}
	
	static void readArray(Serialized &node, Parser &p)
	{
		node.type(Serialized::Array);
		
		NNHardAssert(p.consume('['), "Expected opening bracket!");
		p.consumeWhitespace();
		
		size_t i = 0;
		while(p.peek() != ']')
		{
			if(i > 0)
				NNHardAssert(p.consume(','), "Expected comma or closing bracket!");
			
			p.consumeWhitespace();
			
			Serialized *value = new Serialized();
			readValue(*value, p);
			
			node.add(value);
			p.consumeWhitespace();
			
			++i;
		}
		
		p.ignore();
	}
	
	static void readObject(Serialized &node, Parser &p)
	{
		node.type(Serialized::Object);
		
		NNHardAssert(p.consume('{'), "Expected opening bracket!");
		p.consumeWhitespace();
		
		size_t i = 0;
		while(p.peek() != '}')
		{
			if(i > 0)
				NNHardAssert(p.consume(','), "Expected comma or closing bracket!");
			
			p.consumeWhitespace();
			
			Serialized key;
			readString(key, p);
			p.consumeWhitespace();
			
			NNHardAssert(p.consume(':'), "Expected :!");
			
			Serialized *value = new Serialized();
			readValue(*value, p);
			
			node.set(key.as<std::string>(), value);
			p.consumeWhitespace();
			++i;
		}
		
		p.ignore();
	}
	
// MARK: Writing
	
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
			write(key, out);
			out << ':';
			if(level >= 0)
				out << ' ';
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
