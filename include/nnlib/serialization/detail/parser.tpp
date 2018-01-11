#ifndef SERIALIZATION_PARSER_TPP
#define SERIALIZATION_PARSER_TPP

#include "../parser.hpp"
#include <sstream>

namespace nnlib
{

Parser::Parser(std::istream &in) :
	m_in(in)
{}

bool Parser::eof() const
{
	return m_in.peek() == EOF;
}

char Parser::peek() const
{
	return m_in.peek();
}

char Parser::get()
{
	return m_in.get();
}

void Parser::ignore()
{
	m_in.ignore();
}

void Parser::skipLine()
{
	while(!eof() && peek() != '\n')
		ignore();
	if(!eof())
		ignore();
}

bool Parser::consume(char c)
{
	if(m_in.peek() == c)
	{
		m_in.ignore();
		return true;
	}
	return false;
}

bool Parser::consume(const std::string &sequence)
{
	size_t i = 0, end = sequence.length();
	while(i != end && sequence[i] == m_in.get())
		++i;

	if(i < end)
	{
		while(i > 0)
		{
			m_in.unget();
			--i;
		}

		return false;
	}

	return true;
}

std::string Parser::consumeCombinationOf(const std::string &chars)
{
	std::string result;
	while(chars.find(m_in.peek()) != std::string::npos)
		result.push_back(m_in.get());
	return result;
}

std::string Parser::consumeWhitespace()
{
	return consumeCombinationOf(" \r\n\t");
}

std::string Parser::consumeDigits()
{
	return consumeCombinationOf("0123456789");
}

std::string Parser::consumeUntil(const std::string &chars)
{
	std::string result;
	while(chars.find(m_in.peek()) == std::string::npos && m_in.peek() != EOF)
		result.push_back(m_in.get());
	return result;
}

}

#endif
