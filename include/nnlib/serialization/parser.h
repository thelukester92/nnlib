#ifndef PARSER_H
#define PARSER_H

#include <iostream>
#include <sstream>

namespace nnlib
{

class Parser
{
public:
	Parser(std::istream &in) :
		m_in(&in),
		m_ownsStream(false)
	{}
	
	Parser(const std::string &s) :
		m_in(new std::istringstream(s)),
		m_ownsStream(true)
	{}
	
	Parser(const Parser &) = delete;
	Parser &operator=(const Parser &) = delete;
	
	~Parser()
	{
		if(m_ownsStream)
			delete m_in;
	}
	
	
	
private:
	std::istream *m_in;
	bool m_ownsStream;
};

}

#endif
