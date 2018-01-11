#ifndef SERIALIZATION_FILE_SERIALIZER_TPP
#define SERIALIZATION_FILE_SERIALIZER_TPP

#include <algorithm>
#include "../binaryserializer.hpp"
#include "../fileserializer.hpp"
#include "../csvserializer.hpp"
#include "../jsonserializer.hpp"

namespace nnlib
{

Serialized FileSerializer::read(const std::string &filename)
{
	std::string ext = filename.substr(filename.size() - 4);
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

	if(ext == ".bin")
		return BinarySerializer::read(filename);
	else if(ext == ".csv")
		return CSVSerializer::read(filename);
	else if(ext == "json")
		return JSONSerializer::read(filename);
	else
		throw Error("Serialization failed! Expected bin, csv, or json file extension.");
}

void FileSerializer::write(const Serialized &root, const std::string &filename)
{
	std::string ext = filename.substr(filename.size() - 4);
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

	if(ext == ".bin")
		BinarySerializer::write(root, filename);
	else if(ext == ".csv")
		CSVSerializer::write(root, filename);
	else if(ext == "json")
		JSONSerializer::write(root, filename);
	else
		throw Error("Serialization failed! Expected bin, csv, or json file extension.");
}

}

#endif
