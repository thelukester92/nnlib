#include "../test_jsonserializer.hpp"
#include "nnlib/serialization/jsonserializer.hpp"
#include "nnlib/serialization/serialized.hpp"
#include "nnlib/nn/linear.hpp"
#include "nnlib/nn/sequential.hpp"
#include "nnlib/nn/tanh.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(JSONSerializer)
{
    NNTestMethod(read)
    {
        NNTestParams(std::istream &)
        {
            Sequential<T> nn(new Linear<T>(10, 5), new TanH<T>(), new Linear<T>(5, 10), new TanH<T>());

            Serialized s;
            s.set("null", nullptr);
            s.set("bool", true);
            s.set("int", 32);
            s.set("int", 42);
            s.set("double", 3.14);
            s.set("string", "nnlib");
            s.set("array", Serialized::Array);
            s.get("array")->add("array_element");
            s.set("object", Serialized::Object);
            s.get("object")->set("object_prop1", 3.14);
            s.get("object")->set("object_prop2", "value");
            s.set("nn", nn);

            std::stringstream ss;
            JSONSerializer::write(s, ss);
            Serialized t = JSONSerializer::read(ss);

            NNTestEquals(t.type("null"), Serialized::Null);
            NNTestEquals(t.get<bool>("bool"), true);
            NNTestEquals(t.get<int>("int"), 42);
            NNTestAlmostEquals(t.get<double>("double"), 3.14, 1e-12);
            NNTestEquals(t.get<std::string>("string"), "nnlib");
            NNTestEquals(t.size("array"), 1);
            NNTestEquals(t.get("array")->get<std::string>(0), "array_element");
            NNTestEquals(t.size("object"), 2);
            NNTestAlmostEquals(t.get("object")->get<double>("object_prop1"), 3.14, 1e-12);
            NNTestEquals(t.get("object")->get<std::string>("object_prop2"), "value");

            auto *deserialized = t.get<Sequential<T> *>("nn");
            try
            {
                forEach([&](T orig, T copy)
                {
                    NNTestAlmostEquals(orig, copy, 1e-12);
                }, nn.params(), deserialized->params());
            }
            catch(const Error &e)
            {
                delete deserialized;
                throw e;
            }

            std::stringstream ss2;
            ss2 << "[ 1e-5, 3.1415E3, -1e+04, \"\\\"escaped\\\"\" ]";
            auto u = JSONSerializer::read(ss2);
            NNTestAlmostEquals(u.get<double>(0), 1e-5, 1e-12);
            NNTestAlmostEquals(u.get<double>(1), 3.1415e3, 1e-12);
            NNTestAlmostEquals(u.get<double>(2), -1e4, 1e-12);
            NNTestEquals(u.get<std::string>(3), "\"escaped\"");
        }

        NNTestParams(const std::string &)
        {
            Sequential<T> nn(new Linear<T>(10, 5), new TanH<T>(), new Linear<T>(5, 10), new TanH<T>());

            Serialized s;
            s.set("null", nullptr);
            s.set("bool", false);
            s.set("int", 32);
            s.set("int", 42);
            s.set("double", 3.14);
            s.set("string", "nnlib");
            s.set("array", Serialized::Array);
            s.get("array")->add("array_element");
            s.set("object", Serialized::Object);
            s.get("object")->set("object_prop1", 3.14);
            s.get("object")->set("object_prop2", "value");
            s.set("nn", nn);

            JSONSerializer::write(s, ".nnlib.tmp");

            try
            {
                Serialized t = JSONSerializer::read(".nnlib.tmp");
                NNTestEquals(t.type("null"), Serialized::Null);
                NNTestEquals(t.get<bool>("bool"), false);
                NNTestEquals(t.get<int>("int"), 42);
                NNTestAlmostEquals(t.get<double>("double"), 3.14, 1e-12);
                NNTestEquals(t.get<std::string>("string"), "nnlib");
                NNTestEquals(t.size("array"), 1);
                NNTestEquals(t.get("array")->get<std::string>(0), "array_element");
                NNTestEquals(t.size("object"), 2);
                NNTestAlmostEquals(t.get("object")->get<double>("object_prop1"), 3.14, 1e-12);
                NNTestEquals(t.get("object")->get<std::string>("object_prop2"), "value");

                auto *deserialized = t.get<Sequential<T> *>("nn");
                try
                {
                    forEach([&](T orig, T copy)
                    {
                        NNTestAlmostEquals(orig, copy, 1e-12);
                    }, nn.params(), deserialized->params());
                }
                catch(const Error &e)
                {
                    delete deserialized;
                    throw e;
                }
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
        NNTestParams(const Serialized &, std::ostream &, bool)
        {
            Serialized s;
            s.set("null", nullptr);
            s.set("bool", true);
            s.set("int", 32);
            s.set("int", 42);
            s.set("double", 3.14);
            s.set("string", "nnlib");
            s.set("escaped", "\"escaped\"");
            s.set("array", Serialized::Array);
            s.get("array")->add("array_element");
            s.set("object", Serialized::Object);
            s.get("object")->set("object_prop1", 3.14);
            s.get("object")->set("object_prop2", "value");
            s.set("emptyArray", Serialized::Array);
            s.set("emptyObject", Serialized::Object);

            std::stringstream mini, pretty;
            JSONSerializer::write(s, mini, false);
            JSONSerializer::write(s, pretty, true);

            NNTest(mini.str() == "{\"null\":null,\"bool\":true,\"int\":42,\"double\":3.14,\"string\":\"nnlib\",\"escaped\":\"\\\"escaped\\\"\",\"array\":[\"array_element\"],\"object\":{\"object_prop1\":3.14,\"object_prop2\":\"value\"},\"emptyArray\":[],\"emptyObject\":{}}");
            NNTest(pretty.str() == "{\n\t\"null\": null,\n\t\"bool\": true,\n\t\"int\": 42,\n\t\"double\": 3.14,\n\t\"string\": \"nnlib\",\n\t\"escaped\": \"\\\"escaped\\\"\",\n\t\"array\": [\n\t\t\"array_element\"\n\t],\n\t\"object\": {\n\t\t\"object_prop1\": 3.14,\n\t\t\"object_prop2\": \"value\"\n\t},\n\t\"emptyArray\": [],\n\t\"emptyObject\": {}\n}");
        }
    }
}
