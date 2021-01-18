#include "nlohmann/json.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace nlohmann;
int main() {
    json j;
    j["number"] = 1;
    j["float"] = 1.5;
    j["string"] = "this is a string";
    j["boolean"] = true;
    j["user"]["id"] = 10;
    j["user"]["name"] = "Nomango";
    std::cout << j["user"]["name"] << std::endl;


    // 从文件读取 JSON
    std::ifstream ifs("conf/test.json");
    json j2;
    ifs >> j2;

    std::cout  << j2["global"]["epoch"] << std::endl;
    std::string s = j2["global"]["test_file"];
    std::cout << s << std::endl;

    auto layers = j2["layers"];

    // 使用迭代器遍历

    for (int i =0; i < layers.size(); i++) {
        auto &layer = layers[i];
        for (auto iter = layer.begin(); iter != layer.end(); iter++) {
            std::string tmp = iter.key();
            if (tmp == "name")
                std::cout << iter.key() << ":" << iter.value() << std::endl;
        }
    }

}