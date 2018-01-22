#ifndef SERIALIZATION_CSV_SERIALIZER_TPP
#define SERIALIZATION_CSV_SERIALIZER_TPP

#include "../csvserializer.hpp"
#include <fstream>
#include <limits>
#include <sstream>

namespace nnlib
{

Serialized CSVSerializer::read(std::istream &in, size_t skipLines, char delim)
{
    Parser p(in);
    for(size_t i = 0; i < skipLines; ++i)
        p.skipLine();

    Serialized rows(Serialized::Array);
    Serialized *row = readRow(p, delim);

    while(row != nullptr)
    {
        rows.push(row);
        row = readRow(p, delim);
    }

    return rows;
}

Serialized CSVSerializer::read(const std::string &filename, size_t skipLines, char delim)
{
    std::ifstream fin(filename);
    Serialized result = read(fin, skipLines, delim);
    fin.close();
    return result;
}

void CSVSerializer::write(const Serialized &rows, std::ostream &out, char delim)
{
    NNHardAssertEquals(rows.type(), Serialized::Array, "Expected an array!");
    out.precision(std::numeric_limits<double>::digits10);
    for(size_t i = 0, len = rows.size(); i < len; ++i)
        writeRow(rows.get(i), out, delim);
}

void CSVSerializer::write(const Serialized &rows, const std::string &filename, char delim)
{
    std::ofstream fout(filename);
    write(rows, fout, delim);
    fout.close();
}

Serialized *CSVSerializer::readRow(Parser &p, char delim)
{
    p.consumeWhitespace();
    if(p.peek() == EOF)
        return nullptr;

    Serialized *row = new Serialized(Serialized::Array);

    size_t i = 0;
    while(!p.eof() && p.peek() != '\n')
    {
        if(i > 0)
            NNHardAssert(p.consume(delim), "Expected delimiter!");

        p.consumeWhitespace();
        if(p.peek() == '"')
            row->push(readQuoted(p, delim));
        else
            row->push(readUnquoted(p, delim));

        ++i;
    }

    return row;
}

Serialized *CSVSerializer::readQuoted(Parser &p, char delim)
{
    NNHardAssert(p.consume('"'), "Expected quoted string!");

    std::string value;
    bool ok = false;

    while(!p.eof())
    {
        if(p.consume('"'))
        {
            if(p.consume('"'))
                value.push_back('"');
            else
            {
                ok = true;
                break;
            }
        }
        else
            value.push_back(p.get());
    }

    NNHardAssert(ok, "Expected end quote!");
    return new Serialized(value);
}

Serialized *CSVSerializer::readUnquoted(Parser &p, char delim)
{
    bool couldBeNumber = true, foundDecimal = false;
    std::string value;

    if(p.consume('-'))
        value.push_back('-');

    while(!p.eof() && p.peek() != delim && p.peek() != '\n' && p.peek() != '\r')
    {
        if(p.peek() == '.')
        {
            if(foundDecimal)
                couldBeNumber = false;
            else
                foundDecimal = true;
        }
        else if(p.peek() < '0' || p.peek() > '9')
        {
            couldBeNumber = false;
        }

        value.push_back(p.get());
    }

    couldBeNumber = couldBeNumber && value.length() <= 10;

    if(couldBeNumber && foundDecimal)
        return new Serialized(std::stod(value));
    else if(couldBeNumber)
        return new Serialized(std::stoi(value));
    else
        return new Serialized(value);
}

void CSVSerializer::writeRow(const Serialized &row, std::ostream &out, char delim)
{
    NNHardAssertEquals(row.type(), Serialized::Array, "Expected an array!");
    for(size_t i = 0, len = row.size(); i < len; ++i)
    {
        const Serialized &value = row.get(i);

        if(i > 0)
            out << delim;

        switch(value.type())
        {
        case Serialized::Null:
            out << "null";
            break;
        case Serialized::Boolean:
            out << (value.get<bool>() ? "true" : "false");
            break;
        case Serialized::Integer:
            out << value.get<long long>();
            break;
        case Serialized::Float:
            out << value.get<double>();
            break;
        case Serialized::String:
            writeString(value.get<std::string>(), out, delim);
            break;
        default:
            throw Error("Expected primitive value or string!");
        }
    }
    out << std::endl;
}

void CSVSerializer::writeString(const std::string &str, std::ostream &out, char delim)
{
    if(str.find(delim) != std::string::npos || str.find('"') != std::string::npos)
    {
        out << '"';
        for(char c : str)
        {
            if(c == '"')
                out << '"';
            out << c;
        }
        out << '"';
    }
    else
        out << str;
}

}

#endif
