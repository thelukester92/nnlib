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
            std::istringstream ss("0,3.14,string\na,2");
            Serialized s = CSVSerializer::read(ss);
            NNTestEquals(s.size(), 2);
            NNTestEquals(s.size(0), 3);
            NNTestEquals(s.size(1), 2);
            NNTestEquals(s.get(0)->get<int>(0), 0);
            NNTestEquals(s.get(0)->get<double>(1), 3.14);
            NNTestEquals(s.get(0)->get<std::string>(2), "string");
            NNTestEquals(s.get(1)->get<std::string>(0), "a");
            NNTestEquals(s.get(1)->get<int>(1), 2);
        }

        NNTestParams(const std::string &)
        {
            std::ofstream fout(".nnlib.tmp");
            fout << "0,3.14,string\na,2" << std::flush;
            fout.close();

            try
            {
                Serialized s = CSVSerializer::read(".nnlib.tmp");
                NNTestEquals(s.size(), 2);
                NNTestEquals(s.size(0), 3);
                NNTestEquals(s.size(1), 2);
                NNTestEquals(s.get(0)->get<int>(0), 0);
                NNTestEquals(s.get(0)->get<double>(1), 3.14);
                NNTestEquals(s.get(0)->get<std::string>(2), "string");
                NNTestEquals(s.get(1)->get<std::string>(0), "a");
                NNTestEquals(s.get(1)->get<int>(1), 2);
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
            std::istringstream ss("@arff relation foo\n@attribute bar number\n@attribute baz string\n@data\n0,3.14,string\na,2");
            Serialized s = CSVSerializer::read(ss, 4);
            NNTestEquals(s.size(), 2);
            NNTestEquals(s.size(0), 3);
            NNTestEquals(s.size(1), 2);
            NNTestEquals(s.get(0)->get<int>(0), 0);
            NNTestEquals(s.get(0)->get<double>(1), 3.14);
            NNTestEquals(s.get(0)->get<std::string>(2), "string");
            NNTestEquals(s.get(1)->get<std::string>(0), "a");
            NNTestEquals(s.get(1)->get<int>(1), 2);
        }

        NNTestParams(const std::string &, size_t)
        {
            std::ofstream fout(".nnlib.tmp");
            fout << "@arff relation foo\n@attribute bar number\n@attribute baz string\n@data\n0,3.14,string\na,2" << std::flush;
            fout.close();

            try
            {
                Serialized s = CSVSerializer::read(".nnlib.tmp", 4);
                NNTestEquals(s.size(), 2);
                NNTestEquals(s.size(0), 3);
                NNTestEquals(s.size(1), 2);
                NNTestEquals(s.get(0)->get<int>(0), 0);
                NNTestEquals(s.get(0)->get<double>(1), 3.14);
                NNTestEquals(s.get(0)->get<std::string>(2), "string");
                NNTestEquals(s.get(1)->get<std::string>(0), "a");
                NNTestEquals(s.get(1)->get<int>(1), 2);
            }
            catch(const Error &e)
            {
                remove(".nnlib.tmp");
                throw e;
            }

            remove(".nnlib.tmp");
        }

        NNTestParams(std::istream &, size_t, bool)
        {
            std::istringstream ss("0:3.14:string\na:2");
            Serialized s = CSVSerializer::read(ss, 0, ':');
            NNTestEquals(s.size(), 2);
            NNTestEquals(s.size(0), 3);
            NNTestEquals(s.size(1), 2);
            NNTestEquals(s.get(0)->get<int>(0), 0);
            NNTestEquals(s.get(0)->get<double>(1), 3.14);
            NNTestEquals(s.get(0)->get<std::string>(2), "string");
            NNTestEquals(s.get(1)->get<std::string>(0), "a");
            NNTestEquals(s.get(1)->get<int>(1), 2);
        }

        NNTestParams(const std::string &, size_t, bool)
        {
            std::ofstream fout(".nnlib.tmp");
            fout << "0:3.14:string\na:2" << std::flush;
            fout.close();

            try
            {
                Serialized s = CSVSerializer::read(".nnlib.tmp", 0, ':');
                NNTestEquals(s.size(), 2);
                NNTestEquals(s.size(0), 3);
                NNTestEquals(s.size(1), 2);
                NNTestEquals(s.get(0)->get<int>(0), 0);
                NNTestEquals(s.get(0)->get<double>(1), 3.14);
                NNTestEquals(s.get(0)->get<std::string>(2), "string");
                NNTestEquals(s.get(1)->get<std::string>(0), "a");
                NNTestEquals(s.get(1)->get<int>(1), 2);
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
