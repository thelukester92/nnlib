#ifndef SERIALIZATION_JSON_SERIALIZER_TPP
#define SERIALIZATION_JSON_SERIALIZER_TPP

#include "../jsonserializer.hpp"

namespace nnlib
{

Serialized JSONSerializer::read(std::istream &in)
{
	Parser p(in);
	Serialized root;
	readValue(root, p);
	return root;
}

Serialized JSONSerializer::read(const std::string &filename)
{
	std::ifstream fin(filename);
	Serialized result = read(fin);
	fin.close();
	return result;
}

void JSONSerializer::write(const Serialized &root, std::ostream &out, bool pretty)
{
	out.precision(std::numeric_limits<double>::digits10);
	writeValue(root, out, pretty ? 0 : -1);
}

void JSONSerializer::write(const Serialized &root, const std::string &filename, bool pretty)
{
	std::ofstream fout(filename);
	writeValue(root, fout, pretty);
	fout.close();
}

void JSONSerializer::readValue(Serialized &node, Parser &p)
{
	p.consumeWhitespace();
	NNHardAssert(!p.eof(), "Unexpected end of file!");

	char c = p.peek();
	if(c == 'n')
		readNull(node, p);
	else if(c == 't' || c == 'f')
		readBool(node, p);
	else if((c >= '0' && c <= '9') || c == '-')
		readNumber(node, p);
	else if(c == '"')
		readString(node, p);
	else if(c == '[')
		readArray(node, p);
	else if(c == '{')
		readObject(node, p);
}

void JSONSerializer::readNull(Serialized &node, Parser &p)
{
	NNHardAssert(p.consume("null"), "Expected null!");
	node.type(Serialized::Null);
}

void JSONSerializer::readBool(Serialized &node, Parser &p)
{
	if(p.peek() == 't')
	{
		NNHardAssert(p.consume("true"), "Expected true!");
		node.set(true);
	}
	else
	{
		NNHardAssert(p.consume("false"), "Expected false!");
		node.set(false);
	}
}

void JSONSerializer::readNumber(Serialized &node, Parser &p)
{
	std::string intPart, floatPart;

	if(p.consume('-'))
		intPart.push_back('-');

	if(!p.consume('0'))
		intPart = p.consumeDigits();
	else
		intPart.push_back('0');

	if(p.consume('.'))
		floatPart = p.consumeDigits();

	if(p.consume('e') || p.consume('E'))
	{
		floatPart.push_back('e');

		if(p.consume('-'))
			floatPart.push_back('-');
		else
			p.consume('+');

		floatPart += p.consumeDigits();
	}

	if(floatPart.length() == 0 && intPart.length() <= 10)
		node.set(std::stoi(intPart));
	else
		node.set(std::stod(intPart + '.' + floatPart));
}

void JSONSerializer::readString(Serialized &node, Parser &p)
{
	std::string value;

	p.ignore();
	while(p.peek() != '"')
	{
		NNHardAssertNotEquals(p.peek(), EOF, "Expected closing quotation mark!");

		if(p.consume('\\'))
		{
			NNHardAssertNotEquals(p.peek(), EOF, "Expected escaped character!");
			value.push_back(p.get());
		}
		else
			value.push_back(p.get());
	}
	p.ignore();

	node.set(value);
}

void JSONSerializer::readArray(Serialized &node, Parser &p)
{
	node.type(Serialized::Array);

	NNHardAssert(p.consume('['), "Expected opening bracket!");
	p.consumeWhitespace();

	size_t i = 0;
	while(p.peek() != ']')
	{
		if(i > 0)
			NNHardAssert(p.consume(','), "Expected comma or closing bracket!");

		p.consumeWhitespace();

		Serialized *value = new Serialized();
		readValue(*value, p);

		node.add(value);
		p.consumeWhitespace();

		++i;
	}

	p.ignore();
}

void JSONSerializer::readObject(Serialized &node, Parser &p)
{
	node.type(Serialized::Object);

	NNHardAssert(p.consume('{'), "Expected opening bracket!");
	p.consumeWhitespace();

	size_t i = 0;
	while(p.peek() != '}')
	{
		if(i > 0)
			NNHardAssert(p.consume(','), "Expected comma or closing bracket!");

		p.consumeWhitespace();

		Serialized key;
		readString(key, p);
		p.consumeWhitespace();

		NNHardAssert(p.consume(':'), "Expected :!");

		Serialized *value = new Serialized();
		readValue(*value, p);

		node.set(key.get<std::string>(), value);
		p.consumeWhitespace();
		++i;
	}

	p.ignore();
}

void JSONSerializer::indent(std::ostream &out, int level)
{
	for(int i = 0; i < level; ++i)
		out << "\t";
}

void JSONSerializer::newline(std::ostream &out, int level)
{
	if(level >= 0)
		out << "\n";
}

void JSONSerializer::writeValue(const Serialized &root, std::ostream &out, int level)
{
	switch(root.type())
	{
	case Serialized::Null:
		out << "null";
		break;
	case Serialized::Boolean:
		out << (root.get<bool>() ? "true" : "false");
		break;
	case Serialized::Integer:
		out << root.get<long long>();
		break;
	case Serialized::Float:
		out << root.get<double>();
		break;
	case Serialized::String:
		writeString(root.get<std::string>(), out);
		break;
	case Serialized::Array:
		writeArray(root, out, level);
		break;
	case Serialized::Object:
		writeObject(root, out, level);
		break;
	}
}

void JSONSerializer::writeString(const std::string &str, std::ostream &out)
{
	out << '"';
	for(char c : str)
	{
		if(c == '"' || c == '\\')
			out << "\\";
		out << c;
	}
	out << '"';
}

void JSONSerializer::writeArray(const Serialized &node, std::ostream &out, int level)
{
	if(node.size() == 0)
	{
		out << "[]";
		return;
	}

	out << '[';

	for(size_t i = 0, len = node.size(); i < len; ++i)
	{
		if(i > 0)
			out << ',';

		newline(out, level);
		indent(out, level >= 0 ? level + 1 : level);
		write(*node.get(i), out, level >= 0 ? level + 1 : level);
	}

	newline(out, level);
	indent(out, level);
	out << ']';
}

void JSONSerializer::writeObject(const Serialized &node, std::ostream &out, int level)
{
	if(node.size() == 0)
	{
		out << "{}";
		return;
	}

	out << '{';

	size_t i = 0;
	for(const auto &key : node.keys())
	{
		if(i > 0)
			out << ',';

		newline(out, level);
		indent(out, level >= 0 ? level + 1 : level);
		writeString(key, out);
		out << ':';
		if(level >= 0)
			out << ' ';
		write(*node.get(key), out, level >= 0 ? level + 1 : level);

		++i;
	}

	newline(out, level);
	indent(out, level);
	out << '}';
}

}

#endif
