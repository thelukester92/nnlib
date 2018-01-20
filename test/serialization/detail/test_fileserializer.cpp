#include "../test_fileserializer.hpp"
#include "nnlib/serialization/binaryserializer.hpp"
#include "nnlib/serialization/csvserializer.hpp"
#include "nnlib/serialization/fileserializer.hpp"
#include "nnlib/serialization/jsonserializer.hpp"
#include "nnlib/serialization/serialized.hpp"
#include <cstdio>
#include <fstream>
using namespace nnlib;

NNTestClassImpl(FileSerializer)
{
    NNTestMethod(read)
    {
        NNTestParams(const std::string &)
        {
            Serialized s;
            s.add(Serialized::Array);
            s.get(0)->add(0);
            s.get(0)->add(3.14);
            s.get(0)->add("string");
            s.add(Serialized::Array);
            s.get(1)->add("a,string");
            s.get(1)->add(-2);

            BinarySerializer::write(s, ".nnlib.bin");
            try
            {
                Serialized t = FileSerializer::read(".nnlib.bin");
                NNTestEquals(s.size(), 2);
                NNTestEquals(s.size(0), 3);
                NNTestEquals(s.size(1), 2);
                NNTestEquals(s.get(0)->get<int>(0), 0);
                NNTestEquals(s.get(0)->get<double>(1), 3.14);
                NNTestEquals(s.get(0)->get<std::string>(2), "string");
                NNTestEquals(s.get(1)->get<std::string>(0), "a,string");
                NNTestEquals(s.get(1)->get<int>(1), -2);
            }
            catch(const Error &e)
            {
                remove(".nnlib.bin");
                throw e;
            }
            remove(".nnlib.bin");

            CSVSerializer::write(s, ".nnlib.csv");
            try
            {
                Serialized t = FileSerializer::read(".nnlib.csv");
                NNTestEquals(s.size(), 2);
                NNTestEquals(s.size(0), 3);
                NNTestEquals(s.size(1), 2);
                NNTestEquals(s.get(0)->get<int>(0), 0);
                NNTestEquals(s.get(0)->get<double>(1), 3.14);
                NNTestEquals(s.get(0)->get<std::string>(2), "string");
                NNTestEquals(s.get(1)->get<std::string>(0), "a,string");
                NNTestEquals(s.get(1)->get<int>(1), -2);
            }
            catch(const Error &e)
            {
                remove(".nnlib.csv");
                throw e;
            }
            remove(".nnlib.csv");

            JSONSerializer::write(s, ".nnlib.json");
            try
            {
                Serialized t = FileSerializer::read(".nnlib.json");
                NNTestEquals(s.size(), 2);
                NNTestEquals(s.size(0), 3);
                NNTestEquals(s.size(1), 2);
                NNTestEquals(s.get(0)->get<int>(0), 0);
                NNTestEquals(s.get(0)->get<double>(1), 3.14);
                NNTestEquals(s.get(0)->get<std::string>(2), "string");
                NNTestEquals(s.get(1)->get<std::string>(0), "a,string");
                NNTestEquals(s.get(1)->get<int>(1), -2);
            }
            catch(const Error &e)
            {
                remove(".nnlib.json");
                throw e;
            }
            remove(".nnlib.json");
        }
    }

    NNTestMethod(write)
    {
        NNTestParams(const Serialized &, const std::string &)
        {
            Serialized s;
            s.add(Serialized::Array);
            s.get(0)->add(0);
            s.get(0)->add(3.14);
            s.get(0)->add("string");
            s.add(Serialized::Array);
            s.get(1)->add("a,string");
            s.get(1)->add(-2);

            FileSerializer::write(s, ".nnlib.bin");
            try
            {
                Serialized t = BinarySerializer::read(".nnlib.bin");
                NNTestEquals(s.size(), 2);
                NNTestEquals(s.size(0), 3);
                NNTestEquals(s.size(1), 2);
                NNTestEquals(s.get(0)->get<int>(0), 0);
                NNTestEquals(s.get(0)->get<double>(1), 3.14);
                NNTestEquals(s.get(0)->get<std::string>(2), "string");
                NNTestEquals(s.get(1)->get<std::string>(0), "a,string");
                NNTestEquals(s.get(1)->get<int>(1), -2);
            }
            catch(const Error &e)
            {
                remove(".nnlib.bin");
                throw e;
            }
            remove(".nnlib.bin");

            FileSerializer::write(s, ".nnlib.csv");
            try
            {
                Serialized t = CSVSerializer::read(".nnlib.csv");
                NNTestEquals(s.size(), 2);
                NNTestEquals(s.size(0), 3);
                NNTestEquals(s.size(1), 2);
                NNTestEquals(s.get(0)->get<int>(0), 0);
                NNTestEquals(s.get(0)->get<double>(1), 3.14);
                NNTestEquals(s.get(0)->get<std::string>(2), "string");
                NNTestEquals(s.get(1)->get<std::string>(0), "a,string");
                NNTestEquals(s.get(1)->get<int>(1), -2);
            }
            catch(const Error &e)
            {
                remove(".nnlib.csv");
                throw e;
            }
            remove(".nnlib.csv");

            FileSerializer::write(s, ".nnlib.json");
            try
            {
                Serialized t = JSONSerializer::read(".nnlib.json");
                NNTestEquals(s.size(), 2);
                NNTestEquals(s.size(0), 3);
                NNTestEquals(s.size(1), 2);
                NNTestEquals(s.get(0)->get<int>(0), 0);
                NNTestEquals(s.get(0)->get<double>(1), 3.14);
                NNTestEquals(s.get(0)->get<std::string>(2), "string");
                NNTestEquals(s.get(1)->get<std::string>(0), "a,string");
                NNTestEquals(s.get(1)->get<int>(1), -2);
            }
            catch(const Error &e)
            {
                remove(".nnlib.json");
                throw e;
            }
            remove(".nnlib.json");
        }
    }
}
