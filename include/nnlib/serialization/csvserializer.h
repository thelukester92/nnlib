#ifndef CSV_SERIALIZER_H
#define CSV_SERIALIZER_H

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

#include "parser.h"
#include "serialized.h"

namespace nnlib
{

class CSVSerializer
{
public:
	CSVSerializer() = delete;
	CSVSerializer(const CSVSerializer &) = delete;
	CSVSerializer &operator=(const CSVSerializer &) = delete;
	
	static Serialized read(std::istream &in, char delim = ',')
	{
		Parser p(in);
		
		Serialized rows(Serialized::Array);
		Serialized *row = readRow(p, delim);
		
		while(row != nullptr)
		{
			rows.add(row);
			row = readRow(p, delim);
		}
		
		return rows;
	}
	
	static Serialized readString(const std::string &s)
	{
		istringstream iss(s);
		return read(iss);
	}
	
	static Serialized readFile(const std::string &filename)
	{
		ifstream fin(filename);
		Serialized result = read(fin);
		fin.close();
		return result;
	}
	
	static void write(const Serialized &rows, std::ostream &out, char delim = ',')
	{
		out.precision(std::numeric_limits<double>::digits10);
		for(const Serialized *row : rows.as<SerializedArray>())
			writeRow(*row, out, delim);
	}
	
	static void writeFile(const Serialized &rows, const std::string &filename, char delim = ',')
	{
		ofstream fout(filename);
		write(rows, fout, delim);
		fout.close();
	}
	
private:
// MARK: Reading
	
	static Serialized *readRow(Parser &p, char delim)
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
				row->add(readQuoted(p, delim));
			else
				row->add(readUnquoted(p, delim));
			
			++i;
		}
		
		return row;
	}
	
	static Serialized *readQuoted(Parser &p, char delim)
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
	
	static Serialized *readUnquoted(Parser &p, char delim)
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
		
		if(couldBeNumber && foundDecimal)
			return new Serialized(std::stod(value));
		else if(couldBeNumber)
			return new Serialized(std::stoi(value));
		else
			return new Serialized(value);
	}
	
// MARK: Writing
	
	static void writeRow(const Serialized &row, std::ostream &out, char delim)
	{
		size_t i = 0;
		for(Serialized *value : row.as<SerializedArray>())
		{
			if(i > 0)
				out << delim;
			switch(value->type())
			{
			case Serialized::Null:
				out << "null";
				break;
			case Serialized::Boolean:
				out << (value->as<bool>() ? "true" : "false");
				break;
			case Serialized::Integer:
				out << value->as<int>();
				break;
			case Serialized::Float:
				out << value->as<double>();
				break;
			case Serialized::String:
				writeString(value->as<std::string>(), out, delim);
				break;
			default:
				throw Error("Expected primitive value or string!");
			}
			++i;
		}
		out << std::endl;
	}
	
	static void writeString(const std::string &str, std::ostream &out, char delim)
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
};

}

#endif
