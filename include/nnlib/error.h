#ifndef ERROR_H
#define ERROR_H

#include <iostream>
#include <sstream>
#include <exception>
#include <stdexcept>

namespace nnlib
{

class Error : public std::runtime_error
{
public:
	template <typename ... Ts>
	Error(const std::string &file, const std::string &func, int line, const Ts &...reasons) :
		std::runtime_error(file + ":" + std::to_string(line) + " (" + func + "): " + stringify(reasons...))
	{}
	
private:
	template <typename T, typename ... Ts>
	static std::string stringify(const T &value, const Ts &...values)
	{
		return stringify(value) + stringify(values...);
	}
	
	template <typename T>
	static std::string stringify(const T &value)
	{
		return std::to_string(value);
	}
	
	static std::string stringify(const std::string &value)
	{
		return value;
	}
	
	static std::string stringify(const char *value)
	{
		return std::string(value);
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

/// General asserts that can be optimized out after testing.
#ifndef OPTIMIZE
	#define NNAssert(x, ...) NNHardAssert(x, ##__VA_ARGS__)
#else
	#define NNAssert(x, ...) (void) 0
#endif

/// A few convenient ways to use debug asserts.
#define NNAssertEquals(x, y) NNAssert(x == y, "Unexpected value for ", #x, "! Expected ", y, ", got ", x, ".")
#define NNAssertLessThan(x, y) NNAssert(x < y, "Unexpected value for ", #x, "! Expected < ", y, ", got ", x, ".")
#define NNAssertGreaterThan(x, y) NNAssert(x < y, "Unexpected value for ", #x, "! Expected > ", y, ", got ", x, ".")

}

#endif
