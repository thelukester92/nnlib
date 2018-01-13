#include <stdio.h>
#include "../test_fileserializer.hpp"
#include "nnlib/serialization/fileserializer.hpp"
#include "nnlib/serialization/jsonserializer.hpp"
#include "nnlib/serialization/serialized.hpp"
using namespace nnlib;

void TestFileSerializer()
{
    Serialized s;

    s.add(Serialized::Array);
    s.get(0)->add(3.14);
    s.get(0)->add(-12);
    s.get(0)->add(true);
    s.get(0)->add(false);
    s.get(0)->add(Serialized::Null);
    s.get(0)->add("this is a string");

    s.add(Serialized::Array);
    s.get(1)->add("this is a \"string\"");
    s.get(1)->add("this, is, a, string");
    s.get(1)->add("123.456.789");

    try
    {
        FileSerializer::write(s, "nnlib_test_fileserializer.bin");
        FileSerializer::write(s, "nnlib_test_fileserializer.csv");
        FileSerializer::write(s, "nnlib_test_fileserializer.json");

        auto bin = FileSerializer::read("nnlib_test_fileserializer.bin");
        auto csv = FileSerializer::read("nnlib_test_fileserializer.csv");
        auto json = FileSerializer::read("nnlib_test_fileserializer.json");

        for(const Serialized &d : { bin, csv, json })
        {
            NNAssertAlmostEquals(d.get(0)->get<double>(0), 3.14, 1e-12, "FileSerializer failed!");
            NNAssertEquals(d.get(0)->get<int>(1), -12, "FileSerializer failed!");
            NNAssertEquals(d.get(0)->get<std::string>(2), "true", "FileSerializer failed!");
            NNAssertEquals(d.get(0)->get<std::string>(3), "false", "FileSerializer failed!");
            NNAssertEquals(d.get(0)->get<std::string>(4), "null", "FileSerializer failed!");
            NNAssertEquals(d.get(0)->get<std::string>(5), "this is a string", "FileSerializer failed!");
            NNAssertEquals(d.get(1)->get<std::string>(0), "this is a \"string\"", "FileSerializer failed!");
            NNAssertEquals(d.get(1)->get<std::string>(1), "this, is, a, string", "FileSerializer failed!");
            NNAssertEquals(d.get(1)->get<std::string>(2), "123.456.789", "FileSerializer failed!");
        }

        // incompatibility

        bool ok = false;
        try
        {
            FileSerializer::write(s, "nnlib_test_fileserializer.unknown");
        }
        catch(const Error &)
        {
            ok = true;
        }
        NNAssert(ok, "FileSerializer failed! Allowed writing to an unknown file extension!");

        ok = false;
        try
        {
            rename("nnlib_test_fileserializer.json", "nnlib_test_fileserializer.unknown");
            FileSerializer::read("nnlib_test_fileserializer.unknown");
        }
        catch(const Error &e)
        {
            ok = true;
        }
        NNAssert(ok, "FileSerializer failed! Allowed reading from an unknown file extension!");
    }
    catch(const Error &e)
    {
        remove("nnlib_test_fileserializer.bin");
        remove("nnlib_test_fileserializer.csv");
        remove("nnlib_test_fileserializer.json");
        remove("nnlib_test_fileserializer.unknown");
        throw e;
    }

    remove("nnlib_test_fileserializer.bin");
    remove("nnlib_test_fileserializer.csv");
    remove("nnlib_test_fileserializer.json");
    remove("nnlib_test_fileserializer.unknown");
}
