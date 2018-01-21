#include "../test_parser.hpp"
#include "nnlib/serialization/parser.hpp"
#include <sstream>
using namespace nnlib;

NNTestClassImpl(Parser)
{
    NNTestMethod(eof)
    {
        NNTestParams()
        {
            std::stringstream ss;
            Parser p(ss);
            NNTestEquals(p.eof(), true);
        }
    }

    NNTestMethod(peek)
    {
        NNTestParams()
        {
            std::stringstream ss;
            ss << "h";
            Parser p(ss);
            NNTestEquals(p.peek(), 'h');
            NNTestEquals(p.peek(), 'h');
        }
    }

    NNTestMethod(get)
    {
        NNTestParams()
        {
            std::stringstream ss;
            ss << "hi";
            Parser p(ss);
            NNTestEquals(p.get(), 'h');
            NNTestEquals(p.get(), 'i');
        }
    }

    NNTestMethod(ignore)
    {
        NNTestParams()
        {
            std::stringstream ss;
            ss << "hi";
            Parser p(ss);
            p.ignore();
            NNTestEquals(p.get(), 'i');
        }
    }

    NNTestMethod(skipLine)
    {
        NNTestParams()
        {
            std::stringstream ss;
            ss << "hello\ni";
            Parser p(ss);
            p.skipLine();
            NNTestEquals(p.get(), 'i');
        }
    }

    NNTestMethod(consume)
    {
        NNTestParams(char)
        {
            std::stringstream ss;
            ss << "hi";
            Parser p(ss);
            NNTestEquals(p.consume('h'), true);
            NNTestEquals(p.consume('h'), false);
            NNTestEquals(p.consume('i'), true);
        }

        NNTestParams(const std::string &)
        {
            std::stringstream ss;
            ss << "hello there";
            Parser p(ss);
            NNTestEquals(p.consume("hello "), true);
            NNTestEquals(p.consume("hello "), false);
            NNTestEquals(p.consume("their"), false);
            NNTestEquals(p.consume("the"), true);
            NNTestEquals(p.consume("re"), true);
        }
    }

    NNTestMethod(consumeCombinationOf)
    {
        NNTestParams(const std::string &)
        {
            std::stringstream ss;
            ss << "hello there";
            Parser p(ss);
            NNTestEquals(p.consumeCombinationOf("ehlot "), "hello the");
            NNTestEquals(p.consumeCombinationOf("abcde"), "");
            NNTestEquals(p.consumeCombinationOf("re"), "re");
        }
    }

    NNTestMethod(consumeWhitespace)
    {
        NNTestParams()
        {
            std::stringstream ss;
            ss << " \n\thello\r\n\t     \tthere";
            Parser p(ss);
            NNTestEquals(p.consumeWhitespace(), " \n\t");
            NNTestEquals(p.consume("hello"), true);
            NNTestEquals(p.consumeWhitespace(), "\r\n\t     \t");
            NNTestEquals(p.consume("there"), true);
        }
    }

    NNTestMethod(consumeDigits)
    {
        NNTestParams()
        {
            std::stringstream ss;
            ss << "123 456 7890";
            Parser p(ss);
            NNTestEquals(p.consumeDigits(), "123");
            p.ignore();
            NNTestEquals(p.consumeDigits(), "456");
            p.ignore();
            NNTestEquals(p.consumeDigits(), "7890");
        }
    }

    NNTestMethod(consumeUntil)
    {
        NNTestParams(const std::string &)
        {
            std::stringstream ss;
            ss << "Good morning! My name is Luke. Sayounara!";
            Parser p(ss);
            NNTestEquals(p.consumeUntil("!?."), "Good morning");
            p.ignore();
            NNTestEquals(p.consumeUntil("!?."), " My name is Luke");
            p.ignore();
            NNTestEquals(p.consumeUntil("!?."), " Sayounara");
        }
    }
}
