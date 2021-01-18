#pragma once
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/shared_ptr.hpp>
#include <unordered_map>
#include <boost/variant.hpp>
#include <random>
#include <map>
#include <type_traits>
#include <Eigen/Dense>
#include <functional>
#include <memory>

#include "eigen_func.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include "hash_method.h"
#include "ps/ps.h"

namespace YCDL {
    typedef uint64_t ull;
    using SLOT_ID_FEAS = std::pair<int, std::vector<std::string>>;

    inline nlohmann::json &global_conf() {
        static nlohmann::json conf;
        return conf;
    }

    class Layer {};
}
