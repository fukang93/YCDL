#pragma
#include <string>

namespace YCDL {
    uint64_t BKDRHash(std::string s) {
        uint64_t seed = 131; // 31 131 1313 13131 131313 etc..
        uint64_t hash = 0;

        for (auto str : s) {
            hash = hash * seed + str;
        }
        return (hash & ULLONG_MAX);
    }
}