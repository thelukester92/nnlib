#ifndef SERIALIZATION_PARSER_HPP
#define SERIALIZATION_PARSER_HPP

#include "../core/type.hpp"
#include <iostream>

namespace nnlib
{

class Parser
{
public:
    Parser(std::istream &in);

    Parser(const Parser &) = delete;
    Parser &operator=(const Parser &) = delete;

    bool eof() const;

    char peek() const;
    char get();

    void ignore();
    void skipLine();

    bool consume(char c);
    bool consume(const std::string &sequence);
    std::string consumeCombinationOf(const std::string &chars);
    std::string consumeWhitespace();
    std::string consumeDigits();
    std::string consumeUntil(const std::string &chars);

private:
    std::istream &m_in;
};

}

#if !defined NN_REAL_T && !defined NN_IMPL
    #include "detail/parser.tpp"
#endif

#endif
