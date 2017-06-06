#ifndef ARCHIVE_STRING_H
#define ARCHIVE_STRING_H

#include <string>
#include <sstream>
#include <limits>
#include "archive.h"

namespace nnlib
{

class InputStringArchive : public InputArchive<InputStringArchive>
{
public:
	InputStringArchive(const std::string &str) : InputArchive(this, *(new std::istringstream(str)))
	{}
	
	~InputStringArchive()
	{
		delete &(this->m_in);
	}
	
	template <typename T>
	void process(T &arg)
	{
		m_in >> arg;
	}
};


class OutputStringArchive : public OutputArchive<OutputStringArchive>
{
public:
	OutputStringArchive() : OutputArchive(this, *(new std::ostringstream()))
	{
		precision(std::numeric_limits<double>::digits);
	}
	
	~OutputStringArchive()
	{
		delete &(this->m_out);
	}
	
	void precision(size_t precision)
	{
		m_out.precision(precision);
	}
	
	template <typename T>
	void process(const T &arg)
	{
		m_out << arg << " ";
	}
	
	std::string str()
	{
		return dynamic_cast<std::ostringstream &>(this->m_out).str();
	}
};

}

#endif
