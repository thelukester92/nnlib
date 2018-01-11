#ifndef SERIALIZATION_BINARY_SERIALIZER_HPP
#define SERIALIZATION_BINARY_SERIALIZER_HPP

#include "serialized.hpp"
#include <iostream>

namespace nnlib
{

/// Encode/decode data to/from a platform-dependent binary format.
class BinarySerializer
{
public:
	BinarySerializer() = delete;
	BinarySerializer(const BinarySerializer &) = delete;
	BinarySerializer &operator=(const BinarySerializer &) = delete;

	static Serialized read(std::istream &in);
	static Serialized read(const std::string &filename);

	static void write(const Serialized &root, std::ostream &out);
	static void write(const Serialized &root, const std::string &filename);

private:
	static void read(Serialized &node, std::istream &in);
	static void readString(Serialized &node, std::istream &in);
	static void readArray(Serialized &node, std::istream &in);
	static void readObject(Serialized &node, std::istream &in);

	static void writeType(const Serialized &node, std::ostream &out);
	static void writeString(const std::string &s, std::ostream &out);
	static void writeArray(const Serialized &node, std::ostream &out);
	static void writeObject(const Serialized &node, std::ostream &out);
};

}

#if !defined NN_REAL_T && !defined NN_IMPL
	#include "detail/binaryserializer.tpp"
#endif

#endif
