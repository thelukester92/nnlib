#ifndef UTIL_ARGS_TPP
#define UTIL_ARGS_TPP

#include "../args.hpp"
#include "nnlib/core/error.hpp"
#include <iomanip>
#include <map>

#ifndef NN_DBG
    #define NN_ARGS_EXIT_ON_HELP
#endif

namespace nnlib
{

Args::Args(int argc, const char **argv) :
    m_argc(argc),
    m_argv(argv),
    m_argi(0)
{}

Args &Args::unpop()
{
    NNHardAssert(m_argi > 0, "Cannot unpop from a full argument stack!");
    --m_argi;
    return *this;
}

bool Args::hasNext() const
{
    return m_argi < m_argc;
}

bool Args::ifPop(const std::string &s)
{
    if(hasNext() && s == m_argv[m_argi])
    {
        ++m_argi;
        return true;
    }
    return false;
}

bool Args::nextIsNumber() const
{
    if(!hasNext())
        return false;

    try
    {
        std::stod(m_argv[m_argi]);
        return true;
    }
    catch(const std::invalid_argument &)
    {
        return false;
    }
}

std::string Args::popString()
{
    NNHardAssert(hasNext(), "Attempted to pop from an empty argument stack!");
    return m_argv[m_argi++];
}

double Args::popDouble()
{
    NNHardAssert(nextIsNumber(), "Attempted to pop a string as a number!");
    return std::stod(popString());
}

long long Args::popInt()
{
    NNHardAssert(nextIsNumber(), "Attempted to pop a string as a number!");
    return std::stoi(popString());
}

ArgsParser::ArgsParser(bool help) :
    ArgsParser(help ? 'h' : '\0')
{}

ArgsParser::ArgsParser(char helpOpt, std::string helpLong) :
    m_arrayFirst(true),
    m_helpOpt(helpOpt),
    m_nextUnnamedOpt(0)
{
    if(m_helpOpt != '\0')
        addFlag(m_helpOpt, helpLong);
}

ArgsParser &ArgsParser::addFlag(char opt, std::string longOpt)
{
    addOpt(opt, longOpt);
    m_data[opt].set(false);
    return *this;
}

ArgsParser &ArgsParser::addFlag(std::string longOpt)
{
    return addFlag(++m_nextUnnamedOpt, longOpt);
}

ArgsParser &ArgsParser::addInt(char opt, std::string longOpt)
{
    addOpt(opt, longOpt);
    m_expected[opt] = Type::Integer;
    return *this;
}

ArgsParser &ArgsParser::addInt(char opt, std::string longOpt, long long defaultValue)
{
    addInt(opt, longOpt);
    m_data[opt].set(defaultValue);
    return *this;
}

ArgsParser &ArgsParser::addInt(std::string longOpt)
{
    return addInt(++m_nextUnnamedOpt, longOpt);
}

ArgsParser &ArgsParser::addInt(std::string longOpt, long long defaultValue)
{
    return addInt(++m_nextUnnamedOpt, longOpt, defaultValue);
}

ArgsParser &ArgsParser::addInt()
{
    addArrayOpt(Type::Integer);
    return *this;
}

ArgsParser &ArgsParser::addDouble(char opt, std::string longOpt)
{
    addOpt(opt, longOpt);
    m_expected[opt] = Type::Float;
    return *this;
}

ArgsParser &ArgsParser::addDouble(char opt, std::string longOpt, double defaultValue)
{
    addDouble(opt, longOpt);
    m_data[opt].set(defaultValue);
    return *this;
}

ArgsParser &ArgsParser::addDouble(std::string longOpt)
{
    return addDouble(++m_nextUnnamedOpt, longOpt);
}

ArgsParser &ArgsParser::addDouble(std::string longOpt, double defaultValue)
{
    return addDouble(++m_nextUnnamedOpt, longOpt, defaultValue);
}

ArgsParser &ArgsParser::addDouble()
{
    addArrayOpt(Type::Float);
    return *this;
}

ArgsParser &ArgsParser::addString(char opt, std::string longOpt)
{
    addOpt(opt, longOpt);
    m_expected[opt] = Type::String;
    return *this;
}

ArgsParser &ArgsParser::addString(char opt, std::string longOpt, std::string defaultValue)
{
    addString(opt, longOpt);
    m_data[opt].set(defaultValue);
    return *this;
}

ArgsParser &ArgsParser::addString(std::string longOpt)
{
    return addString(++m_nextUnnamedOpt, longOpt);
}

ArgsParser &ArgsParser::addString(std::string longOpt, std::string defaultValue)
{
    return addString(++m_nextUnnamedOpt, longOpt, defaultValue);
}

ArgsParser &ArgsParser::addString()
{
    addArrayOpt(Type::String);
    return *this;
}

ArgsParser &ArgsParser::parse(int argc, const char **argv, bool popCommand, std::ostream &out)
{
    Args args(argc, argv);

    if(popCommand)
        args.popString();

    if(m_arrayFirst && m_arrayExpected.size() > 0)
    {
        m_arrayData.resize(m_arrayExpected.size());
        for(size_t i = 0; i < m_arrayExpected.size(); ++i)
        {
            NNHardAssert(args.hasNext(), "Unexpected end of arguments!");
            if(m_arrayExpected[i] == Type::Integer)
                m_arrayData[i].set(args.popInt());
            else if(m_arrayExpected[i] == Type::Float)
                m_arrayData[i].set(args.popDouble());
            else if(m_arrayExpected[i] == Type::String)
                m_arrayData[i].set(args.popString());
        }
    }

    while(args.hasNext())
    {
        std::string arg = args.popString(), longOpt;
        char opt;

        if(arg == "--")
        {
            // end of named arguments; skip to array args
            break;
        }
        else if(arg.length() < 2 || arg[0] != '-')
        {
            // not a named option; forward it to array args
            args.unpop();
            break;
        }
        else if(arg.length() == 2)
        {
            // a single option
            opt = arg[1];
            auto i = m_expected.find(opt);
            if(i == m_expected.end())
            {
                // unrecognized option; forward it to array args
                args.unpop();
                break;
            }
        }
        else if(arg[1] == '-')
        {
            // a single long option
            longOpt = std::string(arg.c_str() + 2);
            auto i = m_longToChar.find(longOpt);
            if(i != m_longToChar.end())
                opt = i->second;
            else
            {
                // unrecognized option; forward it to array args
                args.unpop();
                break;
            }
        }
        else
        {
            // a multiple-flag option; may not be forwarded to array args (use -- if needed)
            for(size_t i = 1; i < arg.length() - 1; ++i)
            {
                auto j = m_expected.find(arg[i]);
                NNHardAssert(j != m_expected.end(), "Unexpected argument '" + std::string(1, arg[i]) + "'!");
                NNHardAssert(j->second == Type::Boolean, "Multiple options for a single - must be flags!");
                m_data[arg[i]].set(true);
            }
            opt = arg.back();
        }

        auto i = m_expected.find(opt);
        NNHardAssert(i != m_expected.end(), "Unexpected argument '" + std::string(1, opt) + "'!");

        if(i->second == Type::Boolean)
            m_data[opt].set(true);
        else if(i->second == Type::Integer)
            m_data[opt].set(args.popInt());
        else if(i->second == Type::Float)
            m_data[opt].set(args.popDouble());
        else if(i->second == Type::String)
            m_data[opt].set(args.popString());
    }

    if(!m_arrayFirst && m_arrayExpected.size() > 0)
    {
        m_arrayData.resize(m_arrayExpected.size());
        for(size_t i = 0; i < m_arrayExpected.size(); ++i)
        {
            NNHardAssert(args.hasNext(), "Unexpected end of arguments!");
            if(m_arrayExpected[i] == Type::Integer)
                m_arrayData[i].set(args.popInt());
            else if(m_arrayExpected[i] == Type::Float)
                m_arrayData[i].set(args.popDouble());
            else if(m_arrayExpected[i] == Type::String)
                m_arrayData[i].set(args.popString());
        }
    }

    NNHardAssert(!args.hasNext(), "Unexpected argument '" + args.popString() + "'!");

    if(m_helpOpt != '\0' && getFlag(m_helpOpt))
    {
        printHelp(out);
#ifdef NN_ARGS_EXIT_ON_HELP
        exit(0);
#endif
    }

    return *this;
}

const ArgsParser &ArgsParser::printHelp(std::ostream &out) const
{
    out << std::left;

    if(m_arrayFirst)
    {
        for(auto &p : m_arrayExpected)
        {
            if(p == Type::Integer)
                out << "Int" << std::endl;
            else if(p == Type::Float)
                out << "Float" << std::endl;
            else if(p == Type::String)
                out << "String" << std::endl;
        }
    }

    std::map<std::string, Type> orderedOpts;
    for(auto &p : m_expected)
        orderedOpts.emplace(optName(p.first), p.second);

    for(auto &p : orderedOpts)
    {
        char opt;

        if(p.first.size() == 1)
            opt = p.first[0];
        else
            opt = m_longToChar.at(p.first);

        std::string name;
        if(opt < 32)
            name = "--" + p.first;
        else
        {
            name = std::string("-") + opt;
            if(p.first.size() > 1)
                name += ",--" + p.first;
        }

        out << std::setw(25) << name;

        if(p.second == Type::Boolean)
            out << "Flag";
        else if(p.second == Type::Integer)
            out << "Int";
        else if(p.second == Type::Float)
            out << "Double";
        else if(p.second == Type::String)
            out << "String";

        auto d = m_data.find(opt);
        if(d != m_data.end())
        {
            out << " [value = " << d->second.get<std::string>() << "]";
        }

        out << std::endl;
    }

    if(!m_arrayFirst)
    {
        for(auto &p : m_arrayExpected)
        {
            if(p == Type::Integer)
                out << "Int" << std::endl;
            else if(p == Type::Float)
                out << "Float" << std::endl;
            else if(p == Type::String)
                out << "String" << std::endl;
        }
    }

    return *this;
}

const ArgsParser &ArgsParser::printOpts(std::ostream &out) const
{
    out << std::left;

    if(m_arrayFirst)
    {
        for(size_t i = 0; i < m_arrayExpected.size(); ++i)
        {
            if(m_arrayExpected[i] == Type::Integer)
                out << "Int";
            else if(m_arrayExpected[i] == Type::Float)
                out << "Float";
            else if(m_arrayExpected[i] == Type::String)
                out << "String";

            if(i < m_arrayData.size())
                out << " = " << m_arrayData[i].get<std::string>();

            out << std::endl;
        }
    }

    std::map<std::string, Serialized> orderedOpts;
    for(auto &p : m_data)
        orderedOpts.emplace(optName(p.first), p.second);

    for(auto &p : orderedOpts)
        out << std::setw(20) << p.first << "= " << p.second.get<std::string>() << std::endl;

    if(!m_arrayFirst)
    {
        for(size_t i = 0; i < m_arrayExpected.size(); ++i)
        {
            if(m_arrayExpected[i] == Type::Integer)
                out << "Int";
            else if(m_arrayExpected[i] == Type::Float)
                out << "Float";
            else if(m_arrayExpected[i] == Type::String)
                out << "String";

            if(i < m_arrayData.size())
                out << " = " << m_arrayData[i].get<std::string>();

            out << std::endl;
        }
    }

    return *this;
}

bool ArgsParser::hasOpt(char opt) const
{
    auto i = m_data.find(opt);
    return i != m_data.end();
}

bool ArgsParser::hasOpt(std::string longOpt) const
{
    auto i = m_longToChar.find(longOpt);
    return i != m_longToChar.end() && hasOpt(i->second);
}

std::string ArgsParser::optName(char opt) const
{
    auto i = m_charToLong.find(opt);
    if(i != m_charToLong.end())
        return i->second;
    else
        return std::string(1, opt);
}

bool ArgsParser::getFlag(char opt) const
{
    NNHardAssert(hasOpt(opt), "Attempted to get undefined option '" + optName(opt) + "'!");
    NNHardAssert(m_data.at(opt).type() == Type::Boolean, "Attempted to get an incompatible type!");
    return m_data.at(opt).get<bool>();
}

bool ArgsParser::getFlag(std::string opt) const
{
    auto i = m_longToChar.find(opt);
    NNHardAssert(i != m_longToChar.end(), "Attempted to get undefined option '" + opt + "'!");
    return getFlag(i->second);
}

long long ArgsParser::getInt(char opt) const
{
    NNHardAssert(hasOpt(opt), "Attempted to get undefined option '" + optName(opt) + "'!");
    NNHardAssert(m_data.at(opt).type() == Type::Integer, "Attempted to get an incompatible type!");
    return m_data.at(opt).get<long long>();
}

long long ArgsParser::getInt(std::string opt) const
{
    auto i = m_longToChar.find(opt);
    NNHardAssert(i != m_longToChar.end(), "Attempted to get undefined option '" + opt + "'!");
    return getInt(i->second);
}

double ArgsParser::getDouble(char opt) const
{
    NNHardAssert(hasOpt(opt), "Attempted to get undefined option '" + optName(opt) + "'!");
    NNHardAssert(m_data.at(opt).type() == Type::Float, "Attempted to get an incompatible type!");
    return m_data.at(opt).get<double>();
}

double ArgsParser::getDouble(std::string opt) const
{
    auto i = m_longToChar.find(opt);
    NNHardAssert(i != m_longToChar.end(), "Attempted to get undefined option '" + opt + "'!");
    return getDouble(i->second);
}

std::string ArgsParser::getString(char opt) const
{
    NNHardAssert(hasOpt(opt), "Attempted to get undefined option '" + optName(opt) + "'!");
    NNHardAssert(m_data.at(opt).type() == Type::String, "Attempted to get an incompatible type!");
    return m_data.at(opt).get<std::string>();
}

std::string ArgsParser::getString(std::string opt) const
{
    auto i = m_longToChar.find(opt);
    NNHardAssert(i != m_longToChar.end(), "Attempted to get undefined option '" + opt + "'!");
    return getString(i->second);
}

void ArgsParser::addArrayOpt(Type type)
{
    m_arrayExpected.push_back(type);
}

void ArgsParser::addOpt(char opt, std::string longOpt)
{
    NNHardAssert(m_expected.find(opt) == m_expected.end(), "Attempted to redefine '" + optName(opt) + "'!");
    m_expected[opt] = Type::Boolean;

    if(opt != m_helpOpt && m_arrayExpected.size() == 0)
        m_arrayFirst = false;

    if(longOpt != "")
    {
        NNHardAssert(m_charToLong.find(opt) == m_charToLong.end(), "Attempted to redefine '" + optName(opt) + "'!");
        m_longToChar[longOpt] = opt;
        m_charToLong[opt] = longOpt;
    }
}

}

#endif
