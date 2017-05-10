#ifndef ARGS_H
#define ARGS_H

#include "error.h"
#include <unordered_map>

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
		NNAssert(m_argi > 0, "Cannot unpop from a full argument stack!");
		--m_argi;
		return *this;
	}
	
	bool hasNext()
	{
		return m_argi < m_argc;
	}
	
	bool ifPop(const std::string &s)
	{
		NNAssert(hasNext(), "Attempted to pop from an empty argument stack!");
		if(s == m_argv[m_argi])
		{
			++m_argi;
			return true;
		}
		return false;
	}
	
	bool nextIsNumber()
	{
		NNAssert(hasNext(), "Attempted to use empty argument stack!");
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
		NNAssert(hasNext(), "Attempted to pop from an empty argument stack!");
		return m_argv[m_argi++];
	}
	
	double popDouble()
	{
		NNAssert(nextIsNumber(), "Attempted to pop a string as a number!");
		return std::stod(popString());
	}
	
	int popInt()
	{
		NNAssert(nextIsNumber(), "Attempted to pop a string as a number!");
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
	ArgsParser &addFlag(char opt, std::string longOpt = "")
	{
		addOpt(opt, longOpt);
		return *this;
	}
	
	ArgsParser &addInt(char opt, std::string longOpt = "")
	{
		addOpt(opt, longOpt);
		m_expected[opt].type = Data::Int;
		return *this;
	}
	
	ArgsParser &addDouble(char opt, std::string longOpt = "")
	{
		addOpt(opt, longOpt);
		m_expected[opt].type = Data::Double;
		return *this;
	}
	
	ArgsParser &setString(char opt, std::string longOpt = "")
	{
		addOpt(opt, longOpt);
		m_expected[opt].type = Data::String;
		return *this;
	}
	
	ArgsParser &parse(int argc, const char **argv)
	{
		m_data.clear();
		
		Args args(argc, argv);
		args.popString();
		
		while(args.hasNext())
		{
			NNAssert(!args.nextIsNumber(), "Unexpected number!");
			
			std::string arg = args.popString();
			char opt;
			
			if(arg.length() > 1 && arg[0] == '-' && arg[1] == '-')
			{
				auto i = m_long.find(std::string(arg.c_str() + 2));
				NNAssert(i != m_long.end(), "Unexpected argument!");
				opt = i->second;
			}
			else if(arg.length() > 1 && arg[0] == '-')
			{
				for(size_t i = 1; i < arg.length() - 1; ++i)
				{
					auto j = m_expected.find(arg[i]);
					NNAssert(j != m_expected.end(), "Unexpected argument!");
					NNAssert(j->second.type == Data::Bool, "Multiple options for a single - must be flags!");
					m_data[arg[i]].type = Data::Bool;
					m_data[arg[i]].b = true;
				}
				opt = arg.back();
			}
			
			auto i = m_expected.find(opt);
			NNAssert(i != m_expected.end(), "Unexpected argument!");
			m_data[opt].type = i->second.type;
			
			switch(i->second.type)
			{
			case Data::Bool:
				m_data[opt].b = true;
				break;
			case Data::Int:
				m_data[opt].i = args.popInt();
				break;
			case Data::Double:
				m_data[opt].d = args.popDouble();
				break;
			case Data::String:
				m_data[opt].s = args.popString().c_str();
				break;
			}
		}
		
		return *this;
	}
	
	bool hasOpt(char opt)
	{
		return m_data.find(opt) != m_data.end();
	}
	
	bool hasOpt(std::string longOpt)
	{
		auto i = m_long.find(longOpt);
		return i != m_long.end() && hasOpt(i->second);
	}
	
	int getInt(char opt, int def = 0)
	{
		auto i = m_data.find(opt);
		if(i != m_data.end())
			return i->second.i;
		else
			return def;
	}
	
	double getDouble(char opt, double def = 0)
	{
		auto i = m_data.find(opt);
		if(i != m_data.end())
			return i->second.d;
		else
			return def;
	}
	
	std::string getString(char opt, std::string def = "")
	{
		auto i = m_data.find(opt);
		if(i != m_data.end())
			return i->second.s;
		else
			return def;
	}
	
private:
	struct Data
	{
		enum { Bool, Int, Double, String } type;
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
		NNAssert(m_expected.find(opt) == m_expected.end(), "Cannot redefine a command line option!");
		m_expected[opt].type = Data::Bool;
		m_expected[opt].b = false;
		
		if(longOpt != "")
		{
			NNAssert(m_long.find(longOpt) == m_long.end(), "Cannot redefine a long option!");
			m_long[longOpt] = opt;
		}
	}
	
	std::unordered_map<char, Data> m_expected;
	std::unordered_map<char, Data> m_data;
	std::unordered_map<std::string, char> m_long;
};

}

#endif
