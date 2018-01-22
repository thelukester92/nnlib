#include "../test_csvserializer.hpp"
#include "nnlib/core/tensor.hpp"
#include "nnlib/serialization/csvserializer.hpp"
#include "nnlib/serialization/serialized.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
using namespace nnlib;

NNTestClassImpl(CSVSerializer)
{
    NNTestMethod(read)
    {
        NNTestParams(std::istream &)
        {
            std::istringstream ss("0,3.14,string,123.456.789\n\"a,\"\"string\"\"\",-2");
            Serialized s = CSVSerializer::read(ss);
            NNTestEquals(s.size(), 2);
            NNTestEquals(s.size(0), 4);
            NNTestEquals(s.size(1), 2);
            NNTestEquals(s.get(0).get<int>(0), 0);
            NNTestEquals(s.get(0).get<double>(1), 3.14);
            NNTestEquals(s.get(0).get<std::string>(2), "string");
            NNTestEquals(s.get(0).get<std::string>(3), "123.456.789");
            NNTestEquals(s.get(1).get<std::string>(0), "a,\"string\"");
            NNTestEquals(s.get(1).get<int>(1), -2);
        }

        NNTestParams(const std::string &)
        {
            std::ofstream fout(".nnlib.tmp");
            fout << "0,3.14,string\n\"a,\"\"string\"\"\",-2" << std::flush;
            fout.close();

            try
            {
                Serialized s = CSVSerializer::read(".nnlib.tmp");
                NNTestEquals(s.size(), 2);
                NNTestEquals(s.size(0), 3);
                NNTestEquals(s.size(1), 2);
                NNTestEquals(s.get(0).get<int>(0), 0);
                NNTestEquals(s.get(0).get<double>(1), 3.14);
                NNTestEquals(s.get(0).get<std::string>(2), "string");
                NNTestEquals(s.get(1).get<std::string>(0), "a,\"string\"");
                NNTestEquals(s.get(1).get<int>(1), -2);
            }
            catch(const Error &e)
            {
                remove(".nnlib.tmp");
                throw e;
            }

            remove(".nnlib.tmp");
        }

        NNTestParams(std::istream &, size_t)
        {
            std::istringstream ss("@arff relation foo\n@attribute bar number\n@attribute baz string\n@data\n0,3.14,string\n\"a,\"\"string\"\"\",-2");
            Serialized s = CSVSerializer::read(ss, 4);
            NNTestEquals(s.size(), 2);
            NNTestEquals(s.size(0), 3);
            NNTestEquals(s.size(1), 2);
            NNTestEquals(s.get(0).get<int>(0), 0);
            NNTestEquals(s.get(0).get<double>(1), 3.14);
            NNTestEquals(s.get(0).get<std::string>(2), "string");
            NNTestEquals(s.get(1).get<std::string>(0), "a,\"string\"");
            NNTestEquals(s.get(1).get<int>(1), -2);
        }

        NNTestParams(const std::string &, size_t)
        {
            std::ofstream fout(".nnlib.tmp");
            fout << "@arff relation foo\n@attribute bar number\n@attribute baz string\n@data\n0,3.14,string\n\"a,\"\"string\"\"\",-2" << std::flush;
            fout.close();

            try
            {
                Serialized s = CSVSerializer::read(".nnlib.tmp", 4);
                NNTestEquals(s.size(), 2);
                NNTestEquals(s.size(0), 3);
                NNTestEquals(s.size(1), 2);
                NNTestEquals(s.get(0).get<int>(0), 0);
                NNTestEquals(s.get(0).get<double>(1), 3.14);
                NNTestEquals(s.get(0).get<std::string>(2), "string");
                NNTestEquals(s.get(1).get<std::string>(0), "a,\"string\"");
                NNTestEquals(s.get(1).get<int>(1), -2);
            }
            catch(const Error &e)
            {
                remove(".nnlib.tmp");
                throw e;
            }

            remove(".nnlib.tmp");
        }

        NNTestParams(std::istream &, size_t, char)
        {
            std::istringstream ss("0:3.14:string\n\"a:\"\"string\"\"\":-2");
            Serialized s = CSVSerializer::read(ss, 0, ':');
            NNTestEquals(s.size(), 2);
            NNTestEquals(s.size(0), 3);
            NNTestEquals(s.size(1), 2);
            NNTestEquals(s.get(0).get<int>(0), 0);
            NNTestEquals(s.get(0).get<double>(1), 3.14);
            NNTestEquals(s.get(0).get<std::string>(2), "string");
            NNTestEquals(s.get(1).get<std::string>(0), "a:\"string\"");
            NNTestEquals(s.get(1).get<int>(1), -2);
        }

        NNTestParams(const std::string &, size_t, char)
        {
            std::ofstream fout(".nnlib.tmp");
            fout << "0:3.14:string\n\"a:\"\"string\"\"\":-2" << std::flush;
            fout.close();

            try
            {
                Serialized s = CSVSerializer::read(".nnlib.tmp", 0, ':');
                NNTestEquals(s.size(), 2);
                NNTestEquals(s.size(0), 3);
                NNTestEquals(s.size(1), 2);
                NNTestEquals(s.get(0).get<int>(0), 0);
                NNTestEquals(s.get(0).get<double>(1), 3.14);
                NNTestEquals(s.get(0).get<std::string>(2), "string");
                NNTestEquals(s.get(1).get<std::string>(0), "a:\"string\"");
                NNTestEquals(s.get(1).get<int>(1), -2);
            }
            catch(const Error &e)
            {
                remove(".nnlib.tmp");
                throw e;
            }

            remove(".nnlib.tmp");
        }
    }

    NNTestMethod(write)
    {
        NNTestParams(const Serialized &, std::ostream &)
        {
            Serialized s;
            s.push(Serialized::Array);
            s.get(0).push(0);
            s.get(0).push(3.14);
            s.get(0).push("string");
            s.get(0).push(nullptr);
            s.push(Serialized::Array);
            s.get(1).push("a,\"string\"");
            s.get(1).push(-2);
            s.get(1).push(false);

            std::stringstream ss;
            CSVSerializer::write(s, ss);
            NNTest(ss.str() == "0,3.14,string,null\n\"a,\"\"string\"\"\",-2,false\n");

            try
            {
                Serialized s;
                s.push(Serialized::Array);
                s.get(0).push(Serialized::Object);
                CSVSerializer::write(s, ss);
                NNTest(false);
            }
            catch(const Error &e)
            {}
        }

        NNTestParams(const Serialized &, const std::string &)
        {
            Serialized s;
            s.push(Serialized::Array);
            s.get(0).push(0);
            s.get(0).push(3.14);
            s.get(0).push("string");
            s.push(Serialized::Array);
            s.get(1).push("a,\"string\"");
            s.get(1).push(-2);

            CSVSerializer::write(s, ".nnlib.tmp");

            std::ifstream fin(".nnlib.tmp");
            std::stringstream ss;

            std::string line;
            while(fin)
            {
                std::getline(fin, line);
                if(fin)
                    ss << line << '\n';
            }

            try
            {
                NNTest(ss.str() == "0,3.14,string\n\"a,\"\"string\"\"\",-2\n");
            }
            catch(const Error &e)
            {
                remove(".nnlib.tmp");
                throw e;
            }

            remove(".nnlib.tmp");
        }

        NNTestParams(const Serialized &, std::ostream &, char)
        {
            Serialized s;
            s.push(Serialized::Array);
            s.get(0).push(0);
            s.get(0).push(3.14);
            s.get(0).push("string");
            s.push(Serialized::Array);
            s.get(1).push("a:\"string\"");
            s.get(1).push(-2);

            std::stringstream ss;
            CSVSerializer::write(s, ss, ':');
            NNTest(ss.str() == "0:3.14:string\n\"a:\"\"string\"\"\":-2\n");
        }

        NNTestParams(const Serialized &, const std::string &, char)
        {
            Serialized s;
            s.push(Serialized::Array);
            s.get(0).push(0);
            s.get(0).push(3.14);
            s.get(0).push("string");
            s.push(Serialized::Array);
            s.get(1).push("a:\"string\"");
            s.get(1).push(-2);

            CSVSerializer::write(s, ".nnlib.tmp", ':');

            std::ifstream fin(".nnlib.tmp");
            std::stringstream ss;

            std::string line;
            while(fin)
            {
                std::getline(fin, line);
                if(fin)
                    ss << line << '\n';
            }

            try
            {
                NNTest(ss.str() == "0:3.14:string\n\"a:\"\"string\"\"\":-2\n");
            }
            catch(const Error &e)
            {
                remove(".nnlib.tmp");
                throw e;
            }

            remove(".nnlib.tmp");
        }
    }
}
