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

int Args::popInt()
{
    NNHardAssert(nextIsNumber(), "Attempted to pop a string as a number!");
    return std::stoi(popString());
}

ArgsParser::ArgsParser(bool help) : m_helpOpt(help ? 'h' : '\0'), m_nextUnnamedOpt(0)
{
    if(m_helpOpt != '\0')
        addFlag(m_helpOpt, "help");
}

ArgsParser::ArgsParser(char helpOpt, std::string helpLong) : m_helpOpt(helpOpt), m_nextUnnamedOpt(0)
{
    if(m_helpOpt != '\0')
        addFlag(m_helpOpt, helpLong);
}

ArgsParser &ArgsParser::addFlag(char opt, std::string longOpt)
{
    addOpt(opt, longOpt);
    m_data[opt].type = Type::Bool;
    m_data[opt].b = false;
    return *this;
}

ArgsParser &ArgsParser::addFlag(std::string longOpt)
{
    return addFlag(++m_nextUnnamedOpt, longOpt);
}

ArgsParser &ArgsParser::addInt(char opt, std::string longOpt)
{
    addOpt(opt, longOpt);
    m_expected[opt] = Type::Int;
    return *this;
}

ArgsParser &ArgsParser::addInt(char opt, std::string longOpt, int defaultValue)
{
    addInt(opt, longOpt);
    m_data[opt].type = Type::Int;
    m_data[opt].i = defaultValue;
    return *this;
}

ArgsParser &ArgsParser::addInt(std::string longOpt)
{
    return addInt(++m_nextUnnamedOpt, longOpt);
}

ArgsParser &ArgsParser::addInt(std::string longOpt, int defaultValue)
{
    return addInt(++m_nextUnnamedOpt, longOpt, defaultValue);
}

ArgsParser &ArgsParser::addDouble(char opt, std::string longOpt)
{
    addOpt(opt, longOpt);
    m_expected[opt] = Type::Double;
    return *this;
}

ArgsParser &ArgsParser::addDouble(char opt, std::string longOpt, double defaultValue)
{
    addDouble(opt, longOpt);
    m_data[opt].type = Type::Double;
    m_data[opt].d = defaultValue;
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

ArgsParser &ArgsParser::addString(char opt, std::string longOpt)
{
    addOpt(opt, longOpt);
    m_expected[opt] = Type::String;
    return *this;
}

ArgsParser &ArgsParser::addString(char opt, std::string longOpt, std::string defaultValue)
{
    addString(opt, longOpt);
    m_data[opt].type = Type::String;
    m_stringStorage.push_back(defaultValue);
    m_data[opt].i = m_stringStorage.size() - 1;
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

ArgsParser &ArgsParser::parse(int argc, const char **argv, bool popCommand, std::ostream &out)
{
    Args args(argc, argv);

    if(popCommand)
        args.popString();

    while(args.hasNext())
    {
        NNHardAssert(!args.nextIsNumber(), "Unexpected number!");

        std::string arg = args.popString();
        char opt;

        if(arg.length() > 1 && arg[0] == '-' && arg[1] == '-')
        {
            auto i = m_longToChar.find(std::string(arg.c_str() + 2));
            NNHardAssert(i != m_longToChar.end(), "Unexpected argument '" + std::string(arg.c_str() + 2) + "'!");
            opt = i->second;
        }
        else if(arg.length() > 1 && arg[0] == '-')
        {
            for(size_t i = 1; i < arg.length() - 1; ++i)
            {
                auto j = m_expected.find(arg[i]);
                NNHardAssert(j != m_expected.end(), "Unexpected argument '" + std::string(1, arg[i]) + "'!");
                NNHardAssert(j->second == Type::Bool, "Multiple options for a single - must be flags!");
                m_data[arg[i]].type = Type::Bool;
                m_data[arg[i]].b = true;
            }
            opt = arg.back();
        }

        auto i = m_expected.find(opt);
        NNHardAssert(i != m_expected.end(), "Unexpected argument '" + std::string(1, opt) + "'!");
        m_data[opt].type = i->second;

        switch(i->second)
        {
        case Type::Bool:
            m_data[opt].b = true;
            break;
        case Type::Int:
            m_data[opt].i = args.popInt();
            break;
        case Type::Double:
            m_data[opt].d = args.popDouble();
            break;
        case Type::String:
            m_stringStorage.push_back(args.popString());
            m_data[opt].i = m_stringStorage.size() - 1;
            break;
        }
    }

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

    std::map<std::string, Type> orderedOpts;
    for(auto &p : m_expected)
    {
        orderedOpts.emplace(optName(p.first), p.second);
    }

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

        switch(p.second)
        {
        case Type::Bool:
            out << "Flag";
            break;
        case Type::Int:
            out << "Int";
            break;
        case Type::Double:
            out << "Double";
            break;
        case Type::String:
            out << "String";
            break;
        }

        auto d = m_data.find(opt);
        if(d != m_data.end())
        {
            out << " [value = ";
            switch(d->second.type)
            {
            case Type::Bool:
                out << (d->second.b ? "true" : "false");
                break;
            case Type::Int:
                out << d->second.i;
                break;
            case Type::Double:
                out << d->second.d;
                break;
            case Type::String:
                out << "\"" << m_stringStorage[d->second.i] << "\"";
                break;
            }
            out << "]";
        }

        out << std::endl;
    }

    return *this;
}

const ArgsParser &ArgsParser::printOpts(std::ostream &out) const
{
    out << std::left;

    std::map<std::string, Data> orderedOpts;
    for(auto &p : m_data)
    {
        orderedOpts.emplace(optName(p.first), p.second);
    }

    for(auto &p : orderedOpts)
    {
        out << std::setw(20) << p.first << "= ";
        switch(p.second.type)
        {
        case Type::Bool:
            out << p.second.b;
            break;
        case Type::Int:
            out << p.second.i;
            break;
        case Type::Double:
            out << p.second.d;
            break;
        case Type::String:
            out << m_stringStorage[p.second.i];
            break;
        }
        out << std::endl;
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
    NNHardAssert(m_data.at(opt).type == Type::Bool, "Attempted to get an incompatible type!");
    return m_data.at(opt).b;
}

bool ArgsParser::getFlag(std::string opt) const
{
    auto i = m_longToChar.find(opt);
    NNHardAssert(i != m_longToChar.end(), "Attempted to get undefined option '" + opt + "'!");
    return getFlag(i->second);
}

int ArgsParser::getInt(char opt) const
{
    NNHardAssert(hasOpt(opt), "Attempted to get undefined option '" + optName(opt) + "'!");
    NNHardAssert(m_data.at(opt).type == Type::Int, "Attempted to get an incompatible type!");
    return m_data.at(opt).i;
}

int ArgsParser::getInt(std::string opt) const
{
    auto i = m_longToChar.find(opt);
    NNHardAssert(i != m_longToChar.end(), "Attempted to get undefined option '" + opt + "'!");
    return getInt(i->second);
}

double ArgsParser::getDouble(char opt) const
{
    NNHardAssert(hasOpt(opt), "Attempted to get undefined option '" + optName(opt) + "'!");
    NNHardAssert(m_data.at(opt).type == Type::Double, "Attempted to get an incompatible type!");
    return m_data.at(opt).d;
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
    NNHardAssert(m_data.at(opt).type == Type::String, "Attempted to get an incompatible type!");
    return m_stringStorage[m_data.at(opt).i];
}

std::string ArgsParser::getString(std::string opt) const
{
    auto i = m_longToChar.find(opt);
    NNHardAssert(i != m_longToChar.end(), "Attempted to get undefined option '" + opt + "'!");
    return getString(i->second);
}

void ArgsParser::addOpt(char opt, std::string longOpt)
{
    NNHardAssert(m_expected.find(opt) == m_expected.end(), "Attempted to redefine '" + optName(opt) + "'!");
    m_expected[opt] = Type::Bool;

    if(longOpt != "")
    {
        NNHardAssert(m_charToLong.find(opt) == m_charToLong.end(), "Attempted to redefine '" + optName(opt) + "'!");
        m_longToChar[longOpt] = opt;
        m_charToLong[opt] = longOpt;
    }
}

}

#endif
