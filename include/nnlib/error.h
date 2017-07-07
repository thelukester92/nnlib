#ifndef ERROR_H
#define ERROR_H

#include <iostream>
#include <sstream>
#include <exception>
#include <stdexcept>
#include <type_traits>
#include <cmath>

namespace nnlib
{

class Error : public std::runtime_error
{
public:
	Error(const std::string &reason) :
		std::runtime_error(reason)
	{}
	
	template <typename ... Ts>
	Error(const std::string &file, const std::string &func, int line, const Ts &...reasons) :
		std::runtime_error(file + ":" + std::to_string(line) + " (" + func + "): " + stringify(reasons...))
	{}
	
	template <typename T, typename ... Ts>
	static std::string stringify(const T &value, const Ts &...values)
	{
		return stringify(value) + stringify(values...);
	}
	
	template <typename T>
	static typename std::enable_if<std::is_fundamental<T>::value, std::string>::type stringify(const T &value)
	{
		return std::to_string(value);
	}
	
	template <typename T>
	static typename std::enable_if<!std::is_fundamental<T>::value && !std::is_same<T, std::string>::value, std::string>::type stringify(const T &value)
	{
		return "object";
	}
	
	static std::string stringify(const std::string &value)
	{
		return value;
	}
	
	static std::string stringify(const char *value)
	{
		return std::string(value);
	}
	
	static std::string stringify(std::nullptr_t null)
	{
		return "null";
	}
	
	static std::string stringify()
	{
		return "";
	}
};

/// Check for compiler support.
/// This is here because 1) it causes errors, and 2) this file is included everywhere.
#if __cplusplus < 201103L
	#error C++11 is required! Use -std=c++11 if available.
#endif

/// Asserts that should never be optimized out.
/// These should be reserved for things that are
///   1) outside of our control (i.e. a non-existant file) and
///   2) not in the main training loop.

#define NNHardAssert(x, ...) if(!(x)) throw Error(__FILE__, __func__, __LINE__, ##__VA_ARGS__);

#define NNHardAssertEquals(x, y, ...)								\
	NNHardAssert(													\
		(x) == (y), Error::stringify(__VA_ARGS__),					\
		" Expected ", #x, " == ", #y, ", but ", x, " != ", y, "."	\
	)

#define NNHardAssertAlmostEquals(x, y, eps, ...)					\
	NNHardAssert(													\
		std::fabs((x) - (y)) < eps, Error::stringify(__VA_ARGS__),	\
		" Expected ", #x, " ~= ", #y, ", but ", x, " != ", y, "."	\
	)

#define NNHardAssertNotEquals(x, y, ...)							\
	NNHardAssert(													\
		(x) != (y), Error::stringify(__VA_ARGS__),					\
		" Expected ", #x, " != ", #y, ", but ", x, " == ", y, "."	\
	)

#define NNHardAssertLessThan(x, y, ...)								\
	NNHardAssert(													\
		(x) < (y), Error::stringify(__VA_ARGS__),					\
		" Expected ", #x, " < ", #y, ", but ", x, " >= ", y, "."	\
	)

#define NNHardAssertLessThanOrEquals(x, y, ...)						\
	NNHardAssert(													\
		(x) <= (y), Error::stringify(__VA_ARGS__),					\
		" Expected ", #x, " <= ", #y, ", but ", x, " > ", y, "."	\
	)

#define NNHardAssertGreaterThan(x, y, ...)							\
	NNHardAssert(													\
		(x) > (y), Error::stringify(__VA_ARGS__),					\
		" Expected ", #x, " > ", #y, ", but ", x, " <= ", y, "."	\
	)

#define NNHardAssertGreaterThanOrEquals(x, y, ...)					\
	NNHardAssert(													\
		(x) >= (y), Error::stringify(__VA_ARGS__),					\
		" Expected ", #x, " >= ", #y, ", but ", x, " < ", y, "."	\
	)

/// General asserts that can be optimized out after testing.

#ifndef OPTIMIZE
	#define NNAssert(x, ...)						NNHardAssert(x, ##__VA_ARGS__)
	#define NNAssertEquals(x, y, ...)				NNHardAssertEquals(x, y, ##__VA_ARGS__)
	#define NNAssertAlmostEquals(x, y, eps, ...)	NNHardAssertAlmostEquals(x, y, eps, ##__VA_ARGS__)
	#define NNAssertNotEquals(x, y, ...)			NNHardAssertNotEquals(x, y, ##__VA_ARGS__)
	#define NNAssertLessThan(x, y, ...)				NNHardAssertLessThan(x, y, ##__VA_ARGS__)
	#define NNAssertLessThanOrEquals(x, y, ...)		NNHardAssertLessThanOrEquals(x, y, ##__VA_ARGS__)
	#define NNAssertGreaterThan(x, y, ...)			NNHardAssertGreaterThan(x, y, ##__VA_ARGS__)
	#define NNAssertGreaterThanOrEquals(x, y, ...)	NNHardAssertGreaterThanOrEquals(x, y, ##__VA_ARGS__)
#else
	#define NNAssert(...)						(void) 0
	#define NNAssertEquals(...)					(void) 0
	#define NNAssertAlmostEquals(...)			(void) 0
	#define NNAssertNotEquals(...)				(void) 0
	#define NNAssertLessThan(...)				(void) 0
	#define NNAssertLessThanOrEquals(...)		(void) 0
	#define NNAssertGreaterThan(...)			(void) 0
	#define NNAssertGreaterThanOrEquals(...)	(void) 0
#endif

}

#endif
