#ifndef SERIALIZATION_FILE_SERIALIZER_HPP
#define SERIALIZATION_FILE_SERIALIZER_HPP

#include <iostream>
#include "serialized.hpp"

namespace nnlib
{

/// Encode/decode data to/from a file, automatically determining format.
class FileSerializer
{
public:
	FileSerializer() = delete;
	FileSerializer(const FileSerializer &) = delete;
	FileSerializer &operator=(const FileSerializer &) = delete;

	static Serialized read(const std::string &filename);
	static void write(const Serialized &root, const std::string &filename);
};

}

#if !defined NN_REAL_T && !defined NN_IMPL
	#include "detail/fileserializer.tpp"
#endif

#endif
