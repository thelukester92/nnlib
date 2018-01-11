#ifndef SERIALIZATION_JSON_SERIALIZER_HPP
#define SERIALIZATION_JSON_SERIALIZER_HPP

#include "parser.hpp"
#include "serialized.hpp"
#include <iostream>

namespace nnlib
{

class JSONSerializer
{
public:
	JSONSerializer() = delete;
	JSONSerializer(const JSONSerializer &) = delete;
	JSONSerializer &operator=(const JSONSerializer &) = delete;

	static Serialized read(std::istream &in);
	static Serialized read(const std::string &filename);

	static void write(const Serialized &root, std::ostream &out, bool pretty = false);
	static void write(const Serialized &root, const std::string &filename, bool pretty = false);

private:
	static void readValue(Serialized &node, Parser &p);
	static void readNull(Serialized &node, Parser &p);
	static void readBool(Serialized &node, Parser &p);
	static void readNumber(Serialized &node, Parser &p);
	static void readString(Serialized &node, Parser &p);
	static void readArray(Serialized &node, Parser &p);
	static void readObject(Serialized &node, Parser &p);

	static void indent(std::ostream &out, int level);
	static void newline(std::ostream &out, int level);

	static void writeValue(const Serialized &root, std::ostream &out, int level);
	static void writeString(const std::string &str, std::ostream &out);
	static void writeArray(const Serialized &node, std::ostream &out, int level);
	static void writeObject(const Serialized &node, std::ostream &out, int level);
};

}

#if !defined NN_REAL_T && !defined NN_IMPL
	#include "detail/jsonserializer.tpp"
#endif

#endif
