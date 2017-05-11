#ifndef ARCHIVE_H
#define ARCHIVE_H

#include <iostream>
#include <fstream>
#include <type_traits>

namespace nnlib
{

class Archive
{
public:
	Archive(std::istream *in = &std::cin, std::ostream *out = &std::cout, bool binary = false) :
		m_in(in),
		m_out(out),
		m_binary(binary)
	{}
	
	template <typename T>
	typename std::enable_if<std::is_fundamental<T>::value, Archive>::type &operator<<(const T &x)
	{
		*m_out << x;
		if(!m_binary)
			*m_out << " ";
		return *this;
	}
	
	template <typename T>
	typename std::enable_if<!std::is_fundamental<T>::value, Archive>::type &operator<<(const T &x)
	{
		x.save(*this);
		return *this;
	}
	
	template <typename T>
	typename std::enable_if<std::is_fundamental<T>::value, Archive>::type &operator>>(T &x)
	{
		*m_in >> x;
		return *this;
	}
	
	template <typename T>
	typename std::enable_if<!std::is_fundamental<T>::value, Archive>::type &operator>>(T &x)
	{
		x.load(*this);
		return *this;
	}
	
private:
	std::istream *m_in;
	std::ostream *m_out;
	bool m_binary;
};

}

#endif
