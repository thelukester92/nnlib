#include "../test_csvserializer.hpp"
#include "nnlib/core/tensor.hpp"
#include "nnlib/serialization/csvserializer.hpp"
#include "nnlib/serialization/serialized.hpp"
using namespace nnlib;

void TestCSVSerializer()
{
    // basic serialization

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

        std::stringstream ss;
        CSVSerializer::write(s, ss);

        Serialized d = CSVSerializer::read(ss);

        NNAssertAlmostEquals(d.get(0)->get<double>(0), 3.14, 1e-12, "CSVSerializer failed!");
        NNAssertEquals(d.get(0)->get<int>(1), -12, "CSVSerializer failed!");
        NNAssertEquals(d.get(0)->get<std::string>(2), "true", "CSVSerializer failed!");
        NNAssertEquals(d.get(0)->get<std::string>(3), "false", "CSVSerializer failed!");
        NNAssertEquals(d.get(0)->get<std::string>(4), "null", "CSVSerializer failed!");
        NNAssertEquals(d.get(0)->get<std::string>(5), "this is a string", "CSVSerializer failed!");

        NNAssertEquals(d.get(1)->get<std::string>(0), "this is a \"string\"", "CSVSerializer failed!");
        NNAssertEquals(d.get(1)->get<std::string>(1), "this, is, a, string", "CSVSerializer failed!");
        NNAssertEquals(d.get(1)->get<std::string>(2), "123.456.789", "CSVSerializer failed!");

        {
            std::stringstream ss;
            ss << "this\nis\na\nstring";

            Serialized t = CSVSerializer::read(ss, 1);
            NNAssertEquals(t.get(0)->get<std::string>(0), "is", "CSVSerializer failed to skip lines!");
        }

        bool ok = false;
        Serialized notCompatibleWithCSV(Serialized::Object);
        try
        {
            CSVSerializer::write(notCompatibleWithCSV, ss);
        }
        catch(const Error &)
        {
            ok = true;
        }
        NNAssert(ok, "CSVSerializer failed! Accepted an object instead of an array!");

        ok = false;
        notCompatibleWithCSV.add(Serialized::Object);
        try
        {
            CSVSerializer::write(notCompatibleWithCSV, ss);
        }
        catch(const Error &)
        {
            ok = true;
        }
        NNAssert(ok, "CSVSerializer failed! Accepted an object instead of an array!");

        ok = false;
        notCompatibleWithCSV.get(0)->add(Serialized::Object);
        try
        {
            CSVSerializer::write(notCompatibleWithCSV, ss);
        }
        catch(const Error &)
        {
            ok = true;
        }
        NNAssert(ok, "CSVSerializer failed! Accepted an object instead of a number or string!");
    }

    {
        std::stringstream ss;
        ss << "1,2,3\n4,5,6";

        Tensor<NN_REAL_T> t = CSVSerializer::read(ss);
        NNAssertEquals(t.dims(), 2, "CSVSerializer failed to load a matrix!");
        NNAssertEquals(t(0, 0), 1, "CSVSerializer failed to load a matrix!");
        NNAssertEquals(t(1, 1), 5, "CSVSerializer failed to load a matrix!");
    }
}
