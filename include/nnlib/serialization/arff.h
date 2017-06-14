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
	class Relation
	{
	public:
		Relation() : m_name("untitled")
		{}
		
		template <typename T>
		Relation(const Tensor<T> &matrix) : m_name("untitled")
		{
			NNAssertEquals(matrix.dims(), 2, "Relations are only compatible with matrices!");
			for(size_t i = 0, n = matrix.size(1); i < n; ++i)
				addAttribute("attr" + std::to_string(i), "real");
		}
		
		Relation &setName(std::string name)
		{
			m_name = name;
			return *this;
		}
		
		Relation &addAttribute(std::string name, std::string type)
		{
			m_attrNames.push_back(name);
			m_attrTypes.push_back(type);
			return *this;
		}
		
		size_t attributes()
		{
			return m_attrNames.size();
		}
		
	private:
		std::string m_name;
		Storage<std::string> m_attrNames;
		Storage<std::string> m_attrTypes;
	};
	
	ArffSerializer() = delete;
	
	template <typename T = double>
	static void read(Tensor<T> &matrix, std::istream &in, Relation **rel = nullptr)
	{
		Relation *relation = readRelation(in);
		readData(matrix, in, *relation);
		
		if(rel != nullptr)
			*rel = relation;
		else
			delete rel;
	}
	
	template <typename T = double>
	static void write(const Tensor<T> &matrix, std::ostream &out, const Relation *rel = nullptr)
	{
		const Relation *relation = rel;
		if(relation == nullptr)
			relation = new Relation(matrix);
		
		writeRelation(out, *relation);
		writeData(matrix, out, *relation);
		
		if(rel == nullptr)
			delete relation;
	}
	
private:
	static Relation *readRelation(std::istream &in)
	{
		Relation *rel = new Relation();
		std::string token, type;
		
		readString(in, token);
		toLower(token);
		NNAssertEquals(token, "@relation", "Unexpected token '" + token + "' before @relation!");
		
		readString(in, token, true);
		rel->setName(token);
		
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
			rel->addAttribute(token, type);
		}
		
		return rel;
	}
	
	template <typename T>
	static void readData(Tensor<T> &matrix, std::istream &in, Relation &rel)
	{
		NNAssertGreaterThan(rel.attributes(), 0, "Cannot read data with no attributes!");
		
		Storage<Tensor<T> *> rows;
		std::string token;
		
		try
		{
			while(true)
			{
				Tensor<T> *row = new Tensor<T>(rel.attributes());
				for(T &val : *row)
					readNumber(in, val);
				if(!in.fail())
					rows.push_back(row);
				else
					break;
			}
		}
		catch(const std::invalid_argument &e)
		{
			throw Error(e.what());
		}
		
		matrix = Tensor<T>::flatten(rows).resize(rows.size(), rel.attributes());
		for(Tensor<T> *row : rows)
			delete row;
	}
	
	static void writeRelation(std::ostream &out, const Relation &rel)
	{
		
	}
	
	template <typename T>
	static void writeData(const Tensor<T> &matrix, std::ostream &out, const Relation &rel)
	{
		
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
		while(in >> c)
		{
			if(c == ',' || c == '\n')
				break;
			token.push_back(c);
		}
		if(!in.fail())
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
};

}

#endif
