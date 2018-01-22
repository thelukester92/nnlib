#ifndef SERIALIZATION_BINARY_SERIALIZER_TPP
#define SERIALIZATION_BINARY_SERIALIZER_TPP

#include "../binaryserializer.hpp"
#include <fstream>
#include <limits>
#include <sstream>

namespace nnlib
{

Serialized BinarySerializer::read(std::istream &in)
{
    Serialized root;
    read(root, in);
    return root;
}

Serialized BinarySerializer::read(const std::string &filename)
{
    std::ifstream fin(filename);
    NNHardAssert(fin, "Unable to open file '" + filename + "'!");
    Serialized result = read(fin);
    fin.close();
    return result;
}

void BinarySerializer::write(const Serialized &root, std::ostream &out)
{
    writeType(root, out);

    char c;
    long long i;
    double d;

    switch(root.type())
    {
    case Serialized::Null:
        break;
    case Serialized::Boolean:
        c = root.get<bool>() ? 1 : 0;
        out.write(&c, 1);
        break;
    case Serialized::Integer:
        i = root.get<long long>();
        out.write((const char *) &i, sizeof(long long));
        break;
    case Serialized::Float:
        d = root.get<double>();
        out.write((const char *) &d, sizeof(double));
        break;
    case Serialized::String:
        writeString(root.get<std::string>(), out);
        break;
    case Serialized::Array:
        writeArray(root, out);
        break;
    case Serialized::Object:
        writeObject(root, out);
        break;
    }
}

void BinarySerializer::write(const Serialized &root, const std::string &filename)
{
    std::ofstream fout(filename, std::ofstream::binary);
    NNHardAssert(fout, "Unable to open file '" + filename + "'!");
    write(root, fout);
    fout.close();
}

void BinarySerializer::read(Serialized &node, std::istream &in)
{
    const char TAG_NULL    = 0;
    const char TAG_BOOLEAN = 1;
    const char TAG_INTEGER = 2;
    const char TAG_FLOAT   = 3;
    const char TAG_STRING  = 4;
    const char TAG_ARRAY   = 5;
    const char TAG_OBJECT  = 6;

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

    NNHardAssert(in, "Unexpected end-of-stream!");
}

void BinarySerializer::readString(Serialized &node, std::istream &in)
{
    size_t size;
    in.read((char *) &size, sizeof(size_t));
    NNHardAssert(in, "Unexpected end-of-stream!");

    std::string s(size, '\0');
    in.read(&s[0], size);
    node.set(s);
}

void BinarySerializer::readArray(Serialized &node, std::istream &in)
{
    size_t size;
    in.read((char *) &size, sizeof(size_t));
    NNHardAssert(in, "Unexpected end-of-stream!");

    node.type(Serialized::Array);
    for(size_t i = 0; i < size; ++i)
    {
        Serialized *value = new Serialized();
        read(*value, in);
        node.push(value);
    }
}

void BinarySerializer::readObject(Serialized &node, std::istream &in)
{
    size_t size;
    in.read((char *) &size, sizeof(size_t));
    NNHardAssert(in, "Unexpected end-of-stream!");

    node.type(Serialized::Object);
    for(size_t i = 0; i < size; ++i)
    {
        Serialized key;
        readString(key, in);

        Serialized *value = new Serialized();
        read(*value, in);
        node.set(key.get<std::string>(), value);
    }
}

void BinarySerializer::writeType(const Serialized &node, std::ostream &out)
{
    const char TAG_NULL    = 0;
    const char TAG_BOOLEAN = 1;
    const char TAG_INTEGER = 2;
    const char TAG_FLOAT   = 3;
    const char TAG_STRING  = 4;
    const char TAG_ARRAY   = 5;
    const char TAG_OBJECT  = 6;

    switch(node.type())
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

void BinarySerializer::writeString(const std::string &s, std::ostream &out)
{
    size_t size = s.size();
    out.write((const char *) &size, sizeof(size_t));
    out.write(s.c_str(), size);
}

void BinarySerializer::writeArray(const Serialized &node, std::ostream &out)
{
    size_t size = node.size();
    out.write((const char *) &size, sizeof(size_t));
    for(size_t i = 0; i < size; ++i)
        write(node.get(i), out);
}

void BinarySerializer::writeObject(const Serialized &node, std::ostream &out)
{
    size_t size = node.size();
    out.write((const char *) &size, sizeof(size_t));
    for(const auto &key : node.keys())
    {
        writeString(key, out);
        write(node.get(key), out);
    }
}

}

#endif
