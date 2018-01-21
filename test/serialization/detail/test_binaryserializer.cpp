#include "../test_binaryserializer.hpp"
#include "nnlib/serialization/binaryserializer.hpp"
#include "nnlib/serialization/serialized.hpp"
#include "nnlib/nn/linear.hpp"
#include "nnlib/nn/sequential.hpp"
#include "nnlib/nn/tanh.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(BinarySerializer)
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
            s.get("array")->push("array_element");
            s.set("object", Serialized::Object);
            s.get("object")->set("object_prop1", 3.14);
            s.get("object")->set("object_prop2", "value");
            s.set("nn", nn);

            std::stringstream ss;
            BinarySerializer::write(s, ss);
            Serialized t = BinarySerializer::read(ss);

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
        }

        NNTestParams(const std::string &)
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
            s.get("array")->push("array_element");
            s.set("object", Serialized::Object);
            s.get("object")->set("object_prop1", 3.14);
            s.get("object")->set("object_prop2", "value");
            s.set("nn", nn);

            BinarySerializer::write(s, ".nnlib.tmp");

            try
            {
                Serialized t = BinarySerializer::read(".nnlib.tmp");
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
