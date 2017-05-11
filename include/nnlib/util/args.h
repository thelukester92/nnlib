#ifndef ARGS_H
#define ARGS_H

#include "error.h"
#include <unordered_map>
#include <map>
#include <vector>
#include <iostream>
#include <iomanip>

namespace nnlib
{

class Args
{
public:
	Args(int argc, const char **argv) :
		m_argc(argc),
		m_argv(argv),
		m_argi(0)
	{}
	
	Args &unpop()
	{
		NNHardAssert(m_argi > 0, "Cannot unpop from a full argument stack!");
		--m_argi;
		return *this;
	}
	
	bool hasNext()
	{
		return m_argi < m_argc;
	}
	
	bool ifPop(const std::string &s)
	{
		NNHardAssert(hasNext(), "Attempted to pop from an empty argument stack!");
		if(s == m_argv[m_argi])
		{
			++m_argi;
			return true;
		}
		return false;
	}
	
	bool nextIsNumber()
	{
		NNHardAssert(hasNext(), "Attempted to use empty argument stack!");
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
	
	std::string popString()
	{
		NNHardAssert(hasNext(), "Attempted to pop from an empty argument stack!");
		return m_argv[m_argi++];
	}
	
	double popDouble()
	{
		NNHardAssert(nextIsNumber(), "Attempted to pop a string as a number!");
		return std::stod(popString());
	}
	
	int popInt()
	{
		NNHardAssert(nextIsNumber(), "Attempted to pop a string as a number!");
		return std::stoi(popString());
	}
	
private:
	int m_argc;
	const char **m_argv;
	int m_argi;
};

class ArgsParser
{
public:
	explicit ArgsParser(bool help = true) : m_helpOpt(help ? 'h' : '\0')
	{
		if(m_helpOpt != '\0')
			addFlag(m_helpOpt, "help");
	}
	
	explicit ArgsParser(char helpOpt, std::string helpLong = "help") : m_helpOpt(helpOpt)
	{
		if(m_helpOpt != '\0')
			addFlag(m_helpOpt, helpLong);
	}
	
	ArgsParser &addFlag(char opt, std::string longOpt = "")
	{
		addOpt(opt, longOpt);
		m_data[opt].type = Type::Bool;
		m_data[opt].b = false;
		return *this;
	}
	
	ArgsParser &addInt(char opt, std::string longOpt = "")
	{
		addOpt(opt, longOpt);
		m_expected[opt] = Type::Int;
		return *this;
	}
	
	ArgsParser &addInt(char opt, std::string longOpt, int defaultValue)
	{
		addInt(opt, longOpt);
		m_data[opt].type = Type::Int;
		m_data[opt].i = defaultValue;
		return *this;
	}
	
	ArgsParser &addDouble(char opt, std::string longOpt = "")
	{
		addOpt(opt, longOpt);
		m_expected[opt] = Type::Double;
		return *this;
	}
	
	ArgsParser &addDouble(char opt, std::string longOpt, double defaultValue)
	{
		addDouble(opt, longOpt);
		m_data[opt].type = Type::Double;
		m_data[opt].d = defaultValue;
		return *this;
	}
	
	ArgsParser &addString(char opt, std::string longOpt = "")
	{
		addOpt(opt, longOpt);
		m_expected[opt] = Type::String;
		return *this;
	}
	
	ArgsParser &addString(char opt, std::string longOpt, std::string defaultValue)
	{
		addString(opt, longOpt);
		m_data[opt].type = Type::String;
		m_stringStorage.push_back(defaultValue);
		m_data[opt].s = m_stringStorage.back().c_str();
		return *this;
	}
	
	/// Parses command line arguments.
	/// If -h or --help is present, this prints help and ends the program.
	ArgsParser &parse(int argc, const char **argv)
	{
		Args args(argc, argv);
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
				m_data[opt].s = m_stringStorage.back().c_str();
				break;
			}
		}
		
		if(m_helpOpt != '\0' && getFlag(m_helpOpt))
		{
			printHelp();
			exit(1);
		}
		
		return *this;
	}
	
	ArgsParser &printHelp(std::ostream &out = std::cout)
	{
		out << std::left;
		
		std::map<std::string, Data> orderedOpts;
		for(auto &p : m_data)
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
			
			std::string name = std::string("-") + opt;
			if(p.first.size() > 1)
			 	name += ",--" + p.first;
			
			out << std::setw(25) << name;
			
			switch(p.second.type)
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
			
			if(m_data.find(opt) != m_data.end())
			{
				out << " [value = ";
				switch(p.second.type)
				{
				case Type::Bool:
					out << (p.second.b ? "true" : "false");
					break;
				case Type::Int:
					out << p.second.i;
					break;
				case Type::Double:
					out << p.second.d;
					break;
				case Type::String:
					out << "\"" << p.second.s << "\"";
					break;
				}
				out << "]";
			}
			
			out << std::endl;
		}
		
		return *this;
	}
	
	ArgsParser &printOpts(std::ostream &out = std::cout)
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
				out << p.second.s;
				break;
			}
			out << std::endl;
		}
		
		return *this;
	}
	
	bool hasOpt(char opt)
	{
		auto i = m_data.find(opt);
		return i != m_data.end();
	}
	
	std::string optName(char opt)
	{
		auto i = m_charToLong.find(opt);
		if(i != m_charToLong.end())
			return i->second;
		else
			return std::string(1, opt);
	}
	
	bool getFlag(char opt)
	{
		NNHardAssert(hasOpt(opt), "Attempted to get undefined option '" + optName(opt) + "'!");
		NNHardAssert(m_data.at(opt).type == Type::Bool, "Attempted to get an incompatible type!");
		return m_data.at(opt).b;
	}
	
	bool getFlag(std::string opt)
	{
		auto i = m_longToChar.find(opt);
		NNHardAssert(i != m_longToChar.end(), "Attempted to get undefined option '" + opt + "'!");
		return getFlag(i->second);
	}
	
	int getInt(char opt)
	{
		NNHardAssert(hasOpt(opt), "Attempted to get undefined option '" + optName(opt) + "'!");
		NNHardAssert(m_data.at(opt).type == Type::Int, "Attempted to get an incompatible type!");
		return m_data.at(opt).i;
	}
	
	int getInt(std::string opt)
	{
		auto i = m_longToChar.find(opt);
		NNHardAssert(i != m_longToChar.end(), "Attempted to get undefined option '" + opt + "'!");
		return getInt(i->second);
	}
	
	double getDouble(char opt, double def = 0)
	{
		NNHardAssert(hasOpt(opt), "Attempted to get undefined option '" + optName(opt) + "'!");
		NNHardAssert(m_data.at(opt).type == Type::Double, "Attempted to get an incompatible type!");
		return m_data.at(opt).d;
	}
	
	double getDouble(std::string opt)
	{
		auto i = m_longToChar.find(opt);
		NNHardAssert(i != m_longToChar.end(), "Attempted to get undefined option '" + opt + "'!");
		return getDouble(i->second);
	}
	
	std::string getString(char opt, std::string def = "")
	{
		NNHardAssert(hasOpt(opt), "Attempted to get undefined option '" + optName(opt) + "'!");
		NNHardAssert(m_data.at(opt).type == Type::String, "Attempted to get an incompatible type!");
		return m_data.at(opt).s;
	}
	
	std::string getString(std::string opt)
	{
		auto i = m_longToChar.find(opt);
		NNHardAssert(i != m_longToChar.end(), "Attempted to get undefined option '" + opt + "'!");
		return getString(i->second);
	}
	
private:
	enum class Type { Bool, Int, Double, String };
	struct Data
	{
		Type type;
		union
		{
			bool b;
			int i;
			double d;
			const char *s;
		};
	};
	
	void addOpt(char opt, std::string longOpt)
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
	
	char m_helpOpt;
	std::unordered_map<char, Type> m_expected;
	std::unordered_map<char, Data> m_data;
	std::unordered_map<std::string, char> m_longToChar;
	std::unordered_map<char, std::string> m_charToLong;
	std::vector<std::string> m_stringStorage;
};

}

#endif
