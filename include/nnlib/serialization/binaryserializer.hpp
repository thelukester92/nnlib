#ifndef SERIALIZATION_BINARY_SERIALIZER_HPP
#define SERIALIZATION_BINARY_SERIALIZER_HPP

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

#include "serialized.hpp"

namespace nnlib
{

/// Encode/decode data to/from a platform-dependent binary format.
class BinarySerializer
{
public:
	BinarySerializer() = delete;
	BinarySerializer(const BinarySerializer &) = delete;
	BinarySerializer &operator=(const BinarySerializer &) = delete;
	
	static Serialized read(std::istream &in)
	{
		Serialized root;
		read(root, in);
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
	static void write(const T &value, std::ostream &out)
	{
		write(Serialized(value), out);
	}
	
	static void write(const Serialized &root, std::ostream &out)
	{
		write(root.type(), out);
		
		char c;
		long long i;
		double d;
		
		switch(root.type())
		{
		case Serialized::Null:
			break;
		case Serialized::Boolean:
			c = root.as<bool>() ? 1 : 0;
			out.write(&c, 1);
			break;
		case Serialized::Integer:
			i = root.as<int>();
			out.write((const char *) &i, sizeof(long long));
			break;
		case Serialized::Float:
			d = root.as<double>();
			out.write((const char *) &d, sizeof(double));
			break;
		case Serialized::String:
			write(root.as<std::string>(), out);
			break;
		case Serialized::Array:
			write(root.as<SerializedArray>(), out);
			break;
		case Serialized::Object:
			write(root.as<SerializedObject>(), out);
			break;
		}
	}
	
	template <typename T>
	static void write(const T &value, const std::string &filename)
	{
		writeFile(Serialized(value), filename);
	}
	
	static void writeFile(const Serialized &root, const std::string &filename)
	{
		std::ofstream fout(filename, std::ofstream::binary);
		write(root, fout);
		fout.close();
	}
	
private:
	// MARK: Reading
	
	static void read(Serialized &node, std::istream &in)
	{
		static const char TAG_NULL    = 0;
		static const char TAG_BOOLEAN = 1;
		static const char TAG_INTEGER = 2;
		static const char TAG_FLOAT   = 3;
		static const char TAG_STRING  = 4;
		static const char TAG_ARRAY   = 5;
		static const char TAG_OBJECT  = 6;
		
		char tag;
		in.read(&tag, 1);
		
		char c;
		long long i;
		double d;
		
		switch(tag)
		{
		case TAG_NULL:
			node.type(Serialized::Null);
			break;
		case TAG_BOOLEAN:
			in.read(&c, 1);
			node.set(c == 1);
			break;
		case TAG_INTEGER:
			in.read((char *) &i, sizeof(long long));
			node.set(i);
			break;
		case TAG_FLOAT:
			in.read((char *) &d, sizeof(double));
			node.set(d);
			break;
		case TAG_STRING:
			readString(node, in);
			break;
		case TAG_ARRAY:
			readArray(node, in);
			break;
		case TAG_OBJECT:
			readObject(node, in);
			break;
		}
	}
	
	static void readString(Serialized &node, std::istream &in)
	{
		size_t size;
		in.read((char *) &size, sizeof(size_t));
		
		std::string s(size, '\0');
		in.read(&s[0], size);
		node.set(s);
	}
	
	static void readArray(Serialized &node, std::istream &in)
	{
		size_t size;
		in.read((char *) &size, sizeof(size_t));
		
		node.type(Serialized::Array);
		for(size_t i = 0; i < size; ++i)
		{
			Serialized *value = new Serialized();
			read(*value, in);
			node.add(value);
		}
	}
	
	static void readObject(Serialized &node, std::istream &in)
	{
		size_t size;
		in.read((char *) &size, sizeof(size_t));
		
		node.type(Serialized::Object);
		for(size_t i = 0; i < size; ++i)
		{
			Serialized key;
			readString(key, in);
			
			Serialized *value = new Serialized();
			read(*value, in);
			node.set(key.as<std::string>(), value);
		}
	}
	
	// MARK: Writing
	
	static void write(Serialized::Type type, std::ostream &out)
	{
		static const char TAG_NULL    = 0;
		static const char TAG_BOOLEAN = 1;
		static const char TAG_INTEGER = 2;
		static const char TAG_FLOAT   = 3;
		static const char TAG_STRING  = 4;
		static const char TAG_ARRAY   = 5;
		static const char TAG_OBJECT  = 6;
		
		switch(type)
		{
		case Serialized::Null:
			out.write(&TAG_NULL, 1);
			break;
		case Serialized::Boolean:
			out.write(&TAG_BOOLEAN, 1);
			break;
		case Serialized::Integer:
			out.write(&TAG_INTEGER, 1);
			break;
		case Serialized::Float:
			out.write(&TAG_FLOAT, 1);
			break;
		case Serialized::String:
			out.write(&TAG_STRING, 1);
			break;
		case Serialized::Array:
			out.write(&TAG_ARRAY, 1);
			break;
		case Serialized::Object:
			out.write(&TAG_OBJECT, 1);
			break;
		}
	}
	
	static void write(const std::string &s, std::ostream &out)
	{
		size_t size = s.size();
		out.write((const char *) &size, sizeof(size_t));
		out.write(s.c_str(), size);
	}
	
	static void write(const SerializedArray &arr, std::ostream &out)
	{
		size_t size = arr.size();
		out.write((const char *) &size, sizeof(size_t));
		for(Serialized *node : arr)
			write(*node, out);
	}
	
	static void write(const SerializedObject &obj, std::ostream &out)
	{
		size_t size = obj.size();
		out.write((const char *) &size, sizeof(size_t));
		for(const auto &pair : obj)
		{
			write(pair.first, out);
			write(*pair.second, out);
		}
	}
};

}

#endif
