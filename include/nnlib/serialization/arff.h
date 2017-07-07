#ifndef SERIALIZATION_ARFF_H
#define SERIALIZATION_ARFF_H

#include "../error.h"
#include "../tensor.h"
#include <iostream>
#include <fstream>

namespace nnlib
{

/// \brief Serialize to and from ARFF streams.
///
/// This is a serializer, not an archive, because
/// it works on only one kind of data: matrices.
class ArffSerializer
{
public:
	/// Metadata about a column in a matrix.
	class Attribute
	{
	public:
		Attribute(std::string name = "attr") : m_name(name) {}
		void name(std::string name)			{ m_name = name; }
		std::string name() const			{ return m_name; }
		void addValue(std::string value)	{ m_values.push_back(value); }
		size_t values() const				{ return m_values.size(); }
		std::string value(size_t i) const	{ return m_values[i]; }
	private:
		std::string m_name;
		Storage<std::string> m_values;
	};

	/// Metadata about a matrix.
	class Relation
	{
	public:
		template <typename T>
		Relation(const Tensor<T> &matrix) : m_name("untitled")
		{
			NNHardAssertEquals(matrix.dims(), 2, "Relations are only compatible with matrices!");
			for(size_t i = 0; i < matrix.size(1); ++i)
				addAttribute(Attribute("attr" + std::to_string(i)));
		}
		
		Relation(std::string name = "untitled") : m_name(name) {}
		void name(std::string name)					{ m_name = name; }
		std::string name() const					{ return m_name; }
		void addAttribute(Attribute attr)			{ m_attributes.push_back(attr); }
		size_t attributes() const					{ return m_attributes.size(); }
		const Attribute &attribute(size_t i) const	{ return m_attributes[i]; }
	private:
		Storage<Attribute> m_attributes;
		std::string m_name;
	};
	
	ArffSerializer() = delete;
	
	template <typename T = double>
	static Relation read(Tensor<T> &matrix, std::istream &in)
	{
		Relation relation = readMetadata(in);
		try
		{
			readData(matrix, in, relation);
		}
		catch(const std::invalid_argument &e)
		{
			throw Error(e.what());
		}
		return relation;
	}
	
	template <typename T = double>
	static Relation read(Tensor<T> &matrix, const std::string &filename)
	{
		std::ifstream fin(filename.c_str());
		Relation rel = read(matrix, fin);
		fin.close();
		return rel;
	}
	
	template <typename T = double>
	static void write(const Tensor<T> &matrix, std::ostream &out)
	{
		write(matrix, out, Relation(matrix));
	}
	
	template <typename T = double>
	static void write(const Tensor<T> &matrix, std::ostream &out, const Relation &relation)
	{
		writeMetadata(out, relation);
		writeData(matrix, out, relation);
	}
	
	template <typename T = double>
	static void write(const Tensor<T> &matrix, const std::string &filename)
	{
		std::ofstream fout(filename);
		write(matrix, fout);
		fout.close();
	}
	
	template <typename T = double>
	static void write(const Tensor<T> &matrix, const std::string &filename, const Relation &relation)
	{
		std::ofstream fout(filename);
		write(matrix, fout, relation);
		fout.close();
	}
	
private:
	static Relation readMetadata(std::istream &in)
	{
		Relation relation;
		std::string token;
		
		readToken(in, token);
		NNHardAssert(equalsInsensitive(token, "@relation"), "Invalid arff file! Expected @relation.");
		
		readToken(in, token, true);
		relation.name(token);
		
		while(true)
		{
			Attribute attr;
			
			readToken(in, token);
			if(equalsInsensitive(token, "@data"))
				break;
			NNHardAssert(equalsInsensitive(token, "@attribute"), "Invalid arff file! Expected @attribute.");
			
			readToken(in, token, true);
			attr.name(token);
			
			readToken(in, token);
			if(token[0] == '{')
			{
				for(size_t i = 0; i < token.size(); ++i)
					in.unget();
				while(!in.fail())
				{
					readToken(in, token, true, ",}");
					attr.addValue(token);
					in.unget();
					if(in.peek() == '}')
					{
						in.ignore();
						break;
					}
					in.ignore();
				}
			}
			else
				NNHardAssert(
					equalsInsensitive(token, "real") || equalsInsensitive(token, "numeric") || equalsInsensitive(token, "integer"),
					"Invalid arff file! Expected real|numeric|integer|{enum} attribute type."
				);
			
			relation.addAttribute(attr);
		}
		
		return relation;
	}
	
	template <typename T>
	static void readData(Tensor<T> &matrix, std::istream &in, const Relation &relation)
	{
		Storage<Tensor<T> *> rows;
		Tensor<T> row(relation.attributes());
		
		std::string token;
		size_t attributes = relation.attributes();
		while(true)
		{
			if(in.peek() == EOF)
				break;
			
			for(size_t i = 0; i < attributes; ++i)
			{
				if(relation.attribute(i).values() == 0)
				{
					readToken(in, token, false, "\n,");
					row(i) = std::stod(token);
				}
				else
				{
					bool match = false;
					readToken(in, token, true, "\n,");
					for(size_t j = 0, n = relation.attribute(i).values(); j < n; ++j)
					{
						if(token == relation.attribute(i).value(j))
						{
							row(i) = j;
							match = true;
							break;
						}
					}
					NNHardAssert(match, "Invalid enum value '" + token + "'!");
				}
			}
			
			rows.push_back(new Tensor<T>(row.copy()));
		}
		
		matrix = Tensor<T>::flatten(rows).resize(rows.size(), relation.attributes());
		for(Tensor<T> *row : rows)
			delete row;
	}
	
	static void writeMetadata(std::ostream &out, const Relation &relation)
	{
		// write
		
		out << "@relation ";
		writeToken(out, relation.name());
		out << std::endl;
		
		for(size_t i = 0; i < relation.attributes(); ++i)
		{
			out << "@attribute ";
			writeToken(out, relation.attribute(i).name());
			out << " ";
			if(relation.attribute(i).values() == 0)
				out << "real" << std::endl;
			else
			{
				out << "{";
				for(size_t j = 0; j < relation.attribute(i).values(); ++j)
				{
					if(j > 0)
						out << ",";
					writeToken(out, relation.attribute(i).value(j));
				}
				out << "}" << std::endl;
			}
		}
	}
	
	template <typename T>
	static void writeData(const Tensor<T> &matrix, std::ostream &out, const Relation &relation)
	{
		NNHardAssertEquals(matrix.dims(), 2, "Only matrices are compatible with arff files!");
		NNHardAssertEquals(matrix.size(1), relation.attributes(), "Incompatible relation!");
		
		out << "@data\n";
		size_t attributes = relation.attributes();
		for(size_t i = 0, n = matrix.size(0); i < n; ++i)
		{
			for(size_t j = 0; j < attributes; ++j)
			{
				if(j > 0)
					out << ",";
				if(relation.attribute(j).values() == 0)
					out << matrix(i, j);
				else
				{
					bool match = false;
					for(size_t k = 0, m = relation.attribute(j).values(); k < m; ++k)
					{
						if(k == matrix(i, j))
						{
							match = true;
							writeToken(out, relation.attribute(j).value(k));
							break;
						}
					}
					NNHardAssert(match, "Invalid enum index!");
				}
			}
			out << std::endl;
		}
	}
	
	static bool equalsInsensitive(const std::string &a, const std::string &b)
	{
		if(a.size() != b.size())
			return false;
		for(size_t i = 0, n = a.size(); i < n; ++i)
			if(tolower(a[i]) != tolower(b[i]))
				return false;
		return true;
	}
	
	static void readToken(std::istream &in, std::string &token, bool quoted = false, std::string delims = " \t\n")
	{
		static const std::string WHITESPACE = " \t\n";
		
		token = "";
		while(in.peek() != EOF && token == "")
		{
			while(in.peek() != EOF && WHITESPACE.find(in.peek()) != std::string::npos)
				in.ignore();
			
			char c, quote = '\0';
			bool escaped = false;
			
			c = in.peek();
			if(c == '%')
			{
				std::getline(in, token);
				token = "";
				continue;
			}
			else if(quoted && (c == '"' || c == '\''))
			{
				quote = c;
				in.ignore();
			}
			
			while(in.get(c))
			{
				if(escaped)
					escaped = false;
				else if(c == '\\')
					escaped = true;
				else if((quote != '\0' && c == quote) || (quote == '\0' && delims.find(c) != std::string::npos))
					break;
				if(!escaped)
					token.push_back(c);
			}
			
			if(quote != '\0' && c == quote)
			{
				while(in.peek() != EOF && WHITESPACE.find(in.peek()) != std::string::npos)
					in.ignore();
				if(in.peek() != EOF && delims.find(in.peek()) != std::string::npos)
					in.ignore();
			}
		}
	}
	
	static void writeToken(std::ostream &out, const std::string &token)
	{
		if(token.find_first_of(" \t\n") != std::string::npos)
		{
			out << '"';
			for(const char &c : token)
			{
				if(c == '"' || c == '\\')
					out << '\\';
				out << c;
			}
			out << '"';
		}
		else
			out << token;
	}
};

}

#endif
