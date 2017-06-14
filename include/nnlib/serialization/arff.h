#ifndef SERIALIZATION_ARFF_H
#define SERIALIZATION_ARFF_H

#include "../error.h"
#include <iostream>
#include <sstream>

namespace nnlib
{

/// \brief Serialize to and from ARFF streams.
///
/// This is a serializer, not an archive, because
/// it works on only one kind of data: matrices.
class ArffSerializer
{
public:
	class Attribute
	{
	public:
		Attribute(std::string name = "attr") : m_name(name) {}
		std::string name() const { return m_name; }
		std::string type() const { return "real"; }
	private:
		std::string m_name;
		Storage<std::string> m_values;
	};
	
	class Relation
	{
	public:
		Relation() : m_name("untitled")				{}
		Relation(std::string name) : m_name(name)	{}
		void addAttribute(std::string name)			{ m_attributes.push_back(name); }
		std::string name() const					{ return m_name; }
		size_t attributes() const					{ return m_attributes.size(); }
		const Attribute &attribute(size_t i) const	{ return m_attributes[i]; }
	private:
		std::string m_name;
		Storage<Attribute> m_attributes;
	};
	
	ArffSerializer() = delete;
	
	template <typename T = double>
	static Relation read(Tensor<T> &matrix, std::istream &in)
	{
		Relation relation = readRelation(in);
		readData(matrix, in, relation);
		return relation;
	}
	
	template <typename T = double>
	static void write(const Tensor<T> &matrix, std::ostream &out)
	{
		write(matrix, out, Relation(matrix));
	}
	
	template <typename T = double>
	static void write(const Tensor<T> &matrix, std::ostream &out, const Relation &relation)
	{
		writeRelation(out, relation);
		writeData(matrix, out, relation);
	}
	
private:
	static Relation readRelation(std::istream &in)
	{
		std::string token, type;
		
		readString(in, token);
		toLower(token);
		NNAssertEquals(token, "@relation", "Unexpected token '" + token + "' before @relation!");
		readString(in, token, true);
		
		Relation relation(token);
		
		while(true)
		{
			readString(in, token);
			toLower(token);
			if(token == "@data")
				break;
			NNAssertEquals(token, "@attribute", "Unexpected token '" + token + "' before @data!");
			
			readString(in, token, true);
			readString(in, type);
			toLower(type);
			NNAssert(type == "numeric" || type == "integer" || type == "real", "Unexpected type '" + type + "'!");
			relation.addAttribute(token);
		}
		
		return relation;
	}
	
	template <typename T>
	static void readData(Tensor<T> &matrix, std::istream &in, Relation &relation)
	{
		NNAssertGreaterThan(relation.attributes(), 0, "Cannot read data with no attributes!");
		
		Storage<Tensor<T> *> rows;
		Tensor<T> row;
		
		while(readRow(in, row, ','))
		{
			// skip blank lines
			if(row.size(0) == 0)
				continue;
			
			NNAssert(row.size(0) == relation.attributes(), "Unexpected row size!");
			rows.push_back(new Tensor<T>(row.copy()));
		}
		
		matrix = Tensor<T>::flatten(rows).resize(rows.size(), row.size(0));
		for(Tensor<T> *row : rows)
			delete row;
	}
	
	template <typename T>
	static bool readRow(std::istream &in, Tensor<T> &row, char sep)
	{
		std::string line, value;
		
		if(!std::getline(in, line))
			return false;
		
		if(line == "")
		{
			row.resize(0);
			return true;
		}
		
		std::istringstream ss(line);
		Storage<T> &storage = row.storage().resize(0);
		while(std::getline(ss, value, sep))
			storage.push_back(std::stod(value));
		row.resize(storage.size());
		
		return true;
	}
	
	static void writeRelation(std::ostream &out, const Relation &rel)
	{
		out << "@relation " << quoted(rel.name()) << "\n";
		for(size_t i = 0, n = rel.attributes(); i != n; ++i)
			out << "@attribute " << quoted(rel.attribute(i).name()) << " " << rel.attribute(i).type() << "\n";
		out << "@data\n";
	}
	
	template <typename T>
	static void writeData(const Tensor<T> &matrix, std::ostream &out, const Relation &rel)
	{
		for(size_t i = 0, rows = matrix.size(0); i != rows; ++i)
		{
			for(size_t j = 0, cols = matrix.size(1); j != cols; ++j)
			{
				if(j > 0)
					out << ",";
				out << matrix(i, j);
			}
			out << std::endl;
		}
	}
	
	static void toLower(std::string &s)
	{
		for(char &c : s)
			c = tolower(c);
	}
	
	template <typename T>
	static void readNumber(std::istream &in, T &num)
	{
		std::string token;
		char c;
		while(true)
		{
			c = in.peek();
			if(c == ',' || c == '\n')
				break;
			token.push_back(c);
			in.ignore();
		}
		if(c == ',')
			in.ignore();
		num = std::stod(token);
	}
	
	static void readString(std::istream &in, std::string &s, bool quoted = false)
	{
		while(true)
		{
			char c;
			in >> c;
			
			if(quoted && (c == '"' || c == '\''))
			{
				s = "";
				char quote = c;
				bool escaped = false;
				while(in.get(c))
				{
					if(escaped)
						escaped = false;
					else if(c == '\\')
						escaped = true;
					else if(c == quote)
						break;
					if(!escaped)
						s.push_back(c);
				}
			}
			else
			{
				in.unget();
				in >> s;
			}
			
			if(s.size() > 0 && s[0] == '%')
				std::getline(in, s);
			else
				break;
		}
	}
	
	static std::string quoted(const std::string &s)
	{
		if(s.find_first_of("\t, ") != std::string::npos)
		{
			std::string result = "\"";
			for(char c : s)
			{
				if(c == '"' || c == '\\')
					result.push_back('\\');
				result.push_back(c);
			}
			result.push_back('"');
			return result;
		}
		return s;
	}
};

}

#endif
