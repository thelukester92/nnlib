#ifndef SERIALIZATION_PARSER_HPP
#define SERIALIZATION_PARSER_HPP

#include <iostream>
#include <sstream>

namespace nnlib
{

class Parser
{
public:
	Parser(std::istream &in) :
		m_in(in)
	{}
	
	Parser(const Parser &) = delete;
	Parser &operator=(const Parser &) = delete;
	
	bool eof() const
	{
		return m_in.peek() == EOF;
	}
	
	char peek() const
	{
		return m_in.peek();
	}
	
	char get()
	{
		return m_in.get();
	}
	
	void ignore()
	{
		m_in.ignore();
	}
	
	void skipLine()
	{
		while(!eof() && peek() != '\n')
			ignore();
		if(!eof())
			ignore();
	}
	
	bool consume(char c)
	{
		if(m_in.peek() == c)
		{
			m_in.ignore();
			return true;
		}
		return false;
	}
	
	bool consume(const std::string &sequence)
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
	
	std::string consumeCombinationOf(const std::string &chars)
	{
		std::string result;
		while(chars.find(m_in.peek()) != std::string::npos)
			result.push_back(m_in.get());
		return result;
	}
	
	std::string consumeWhitespace()
	{
		return consumeCombinationOf(" \r\n\t");
	}
	
	std::string consumeDigits()
	{
		return consumeCombinationOf("0123456789");
	}
	
	std::string consumeUntil(const std::string &chars)
	{
		std::string result;
		while(chars.find(m_in.peek()) == std::string::npos && m_in.peek() != EOF)
			result.push_back(m_in.get());
		return result;
	}
	
private:
	std::istream &m_in;
};

}

#endif
