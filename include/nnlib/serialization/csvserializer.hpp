#ifndef SERIALIZATION_CSV_SERIALIZER_HPP
#define SERIALIZATION_CSV_SERIALIZER_HPP

#include "parser.hpp"
#include "serialized.hpp"
#include <iostream>

namespace nnlib
{

class CSVSerializer
{
public:
    CSVSerializer() = delete;
    CSVSerializer(const CSVSerializer &) = delete;
    CSVSerializer &operator=(const CSVSerializer &) = delete;

    static Serialized read(std::istream &in, size_t skipLines = 0, char delim = ',');
    static Serialized read(const std::string &filename, size_t skipLines = 0, char delim = ',');

    static void write(const Serialized &rows, std::ostream &out, char delim = ',');
    static void write(const Serialized &rows, const std::string &filename, char delim = ',');

private:
    static Serialized *readRow(Parser &p, char delim);
    static Serialized *readQuoted(Parser &p, char delim);
    static Serialized *readUnquoted(Parser &p, char delim);

    static void writeRow(const Serialized &row, std::ostream &out, char delim);
    static void writeString(const std::string &str, std::ostream &out, char delim);
};

}

#if !defined NN_REAL_T && !defined NN_IMPL
    #include "detail/csvserializer.tpp"
#endif

#endif
