#ifndef UTIL_ARGS_HPP
#define UTIL_ARGS_HPP

#include "../serialization/serialized.hpp"
#include <unordered_map>
#include <vector>
#include <iostream>

namespace nnlib
{

class Args
{
public:
    Args(int argc, const char **argv);

    Args &unpop();
    bool hasNext() const;
    bool ifPop(const std::string &s);
    bool nextIsNumber() const;

    std::string popString();
    double popDouble();
    long long popInt();

private:
    int m_argc;
    const char **m_argv;
    int m_argi;
};

class ArgsParser
{
public:
    using Type = Serialized::Type;

    explicit ArgsParser(bool help = true);
    explicit ArgsParser(char helpOpt, std::string helpLong = "help");

    ArgsParser &addFlag(char opt, std::string longOpt = "");
    ArgsParser &addFlag(std::string longOpt);

    ArgsParser &addInt(char opt, std::string longOpt = "");
    ArgsParser &addInt(char opt, std::string longOpt, long long defaultValue);
    ArgsParser &addInt(std::string longOpt);
    ArgsParser &addInt(std::string longOpt, long long defaultValue);

    ArgsParser &addDouble(char opt, std::string longOpt = "");
    ArgsParser &addDouble(char opt, std::string longOpt, double defaultValue);
    ArgsParser &addDouble(std::string longOpt);
    ArgsParser &addDouble(std::string longOpt, double defaultValue);

    ArgsParser &addString(char opt, std::string longOpt = "");
    ArgsParser &addString(char opt, std::string longOpt, std::string defaultValue);
    ArgsParser &addString(std::string longOpt);
    ArgsParser &addString(std::string longOpt, std::string defaultValue);

    /// \brief Parses command line arguments.
    ///
    /// If -h or --help is present, this prints help and ends the program.
    ArgsParser &parse(int argc, const char **argv, bool popCommand = true, std::ostream &out = std::cout);

    /// Simply prints help (the list of options and their default values, if any).
    const ArgsParser &printHelp(std::ostream &out = std::cout) const;

    /// Prints the list of options and their values.
    const ArgsParser &printOpts(std::ostream &out = std::cout) const;

    bool hasOpt(char opt) const;
    bool hasOpt(std::string longOpt) const;

    std::string optName(char opt) const;

    bool getFlag(char opt) const;
    bool getFlag(std::string opt) const;
    long long getInt(char opt) const;
    long long getInt(std::string opt) const;
    double getDouble(char opt) const;
    double getDouble(std::string opt) const;
    std::string getString(char opt) const;
    std::string getString(std::string opt) const;

private:
    void addOpt(char opt, std::string longOpt);

    char m_helpOpt, m_nextUnnamedOpt;
    std::unordered_map<char, Serialized::Type> m_expected;
    std::unordered_map<char, Serialized> m_data;
    std::unordered_map<std::string, char> m_longToChar;
    std::unordered_map<char, std::string> m_charToLong;
};

}

#if !defined NN_REAL_T && !defined NN_IMPL
    #include "detail/args.tpp"
#endif

#endif
