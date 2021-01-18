#pragma once
#include "all.h"
#include "tools.h"
#include "instance.h"
#include <generator.h>

namespace YCDL {
    Generator<std::vector<Instance> >
    dataload(std::string path, int epoch, int batch_size, bool is_train, int shuffle_num = 1) {
        return Generator < std::vector < Instance > > ([=](Yield <std::vector<Instance>> &yield) {
            std::string s;
            int cnt = 0, pool_sz = shuffle_num * batch_size;
            std::vector<Instance> instances, ans;
            int nums = epoch;
            while (nums) {
                std::ifstream file;
                file.open(path);
                nums--;
                while (getline(file, s)) {
                    cnt += 1;
                    std::vector<std::string> segs;
                    line_split(s, "\t", segs);
                    Instance ins;
                    ins.label = boost::lexical_cast<int>(segs[0]);
                    for (int i = 1; i < segs.size(); i++) {
                        ins.feas.push_back(boost::lexical_cast<ull>(segs[i]));
                        ins.vals.push_back(1.0);
                    }
                    ins.feas.push_back(0);
                    ins.vals.push_back(1.0);
                    instances.emplace_back(std::move(ins));
                    if (cnt == pool_sz) {
                        cnt = 0;
                        if (is_train) {
                            std::random_shuffle(instances.begin(), instances.end());
                        }
                        for (int i = 0; i < pool_sz; i += batch_size) {
                            ans.assign(instances.begin() + i, instances.begin() + i + batch_size);
                            yield(ans);
                        }
                        instances.clear();
                    }
                }
                file.close();
            }
            if (cnt != pool_sz && instances.size() != 0) {
                if (is_train) {
                    std::random_shuffle(instances.begin(), instances.end());
                }
                for (int i = 0; i < cnt; i += batch_size) {
                    if (i + batch_size < cnt) {
                        ans.assign(instances.begin() + i, instances.begin() + i + batch_size);
                    } else {
                        ans.clear();
                        ans.assign(instances.begin() + i, instances.end());
                    }
                    yield(ans);
                }
                instances.clear();
            }
        });
    }
}
