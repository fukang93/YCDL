#include "all.h"
#include "tools.h"
#include "network.h"
#include "ioc.h"
#include "init.h"
#include "instance.h"
#include <generator.h>
using namespace std;

YCDL::ull deal_num (std::string s, int delim, std::string head) {
    std::string t = boost::lexical_cast<std::string>(boost::lexical_cast<int>(s) / delim);
    return YCDL::BKDRHash(head+t);
}

YCDL::ull deal_category(std::string s, std::string head) {
    return YCDL::BKDRHash(head+s);
}

YCDL::ull deal_label(std::string s) {
    if (s == "\"yes\"")
        return 1;
    return 0;
}

YCDL::Generator<std::vector<YCDL::Instance> > dataload(std::string path, int epoch, int batch_size, bool is_train, int shuffle_num = 1) {
    return YCDL::Generator<std::vector<YCDL::Instance> >([=](YCDL::Yield<std::vector<YCDL::Instance> > &yield) {
        std::string s;
        int cnt = 0, pool_sz = shuffle_num * batch_size;
        std::vector<YCDL::Instance> instances, ans;
        int nums = epoch;
        std::vector<std::string> heads = {"age","job","marital","education","default","balance","housing","loan","contact","day","month",
                                          "duration","campaign","pdays","previous","poutcome","y"};
        std::vector<std::string> ways = {"num-10","cat","cat","cat","cat","num-10","cat","cat","cat","num-10","cat","num-10","num-10","num-100","num-1","cat","label"};

        while(nums) {
            std::ifstream file;
            file.open(path);
            nums--;
            while (getline(file, s)) {
                cnt += 1;
                std::vector<std::string> segs;
                YCDL::line_split(s, ";", segs);
                YCDL::Instance ins;
                int len = segs.size();
                for (int i = 0; i < len - 1; i++) {
                    YCDL::ull value;
                    if (ways[i] == "cat") {
                        value = deal_category(segs[i], heads[i]);
                    } else if (ways[i].substr(0,3) == "num") {
                        int delim = boost::lexical_cast<int>(ways[i].substr(4));
                        value = deal_num(segs[i], delim, heads[i]);
                    }
                    ins.slot_feas.push_back({i, vector<string>(1, boost::lexical_cast<string>(value))});
                    //ins.feas.push_back(value);
                    //ins.vals.push_back(1.0);
                }
                //ins.feas.push_back(0);
                //ins.vals.push_back(1.0);
                ins.label = boost::lexical_cast<int>(deal_label(segs[len-1]));

                instances.emplace_back(std::move(ins));
                if (cnt == pool_sz) {
                    cnt = 0;
                    if (is_train) {
                        random_shuffle(instances.begin(), instances.end());
                    }
                    for (int i = 0; i < pool_sz; i+=batch_size) {
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
                random_shuffle(instances.begin(), instances.end());
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

int main() {
    YCDL::initialize();

    nlohmann::json& conf = YCDL::global_conf();
    std::ifstream ifs("conf/nn_config.json");
    ifs >> conf;

    int epoch = conf["epoch"];
    std::string train_file = conf["train_file"];
    std::string test_file = conf["test_file"];
    int batch_size = conf["batch_size"];
    int shuffle_num = conf["shuffle_num"];
    std::string optimizer_name = conf["optimizer"]["name"];
    std::shared_ptr<YCDL::Optimizer> optimizer = YCDL::MakeLayer<YCDL::Optimizer>(optimizer_name);
    YCDL::Network network = YCDL::Network(optimizer);

    int total_num = YCDL::get_file_len(train_file);
    int epoch_batch_num = total_num / batch_size;
    int iter = 0;

    auto data_iter = dataload(train_file, epoch, batch_size, true, shuffle_num);
    for (std::vector<YCDL::Instance> instances : data_iter) {
        iter++;
        //cout << "print weight\n";
        network.forward(instances);
        //optimizer->print_weight();
        network.backward(instances);
        // test auc
        std::vector<YCDL::Instance> train_ins, test_ins;
        int epoch_num = iter / epoch_batch_num;
        if (iter % epoch_batch_num == 0) {
            std::cout << "epoch_num: " << epoch_num << std::endl;
            std::cout << "train:\n";
            auto data_iter_test = dataload(train_file, 1, 100, false);
            for (std::vector<YCDL::Instance> instances: data_iter_test) {
                if (train_ins.size() > 100000) break;
                for (auto x: instances) {
                    train_ins.emplace_back(std::move(x));
                }
            }
            network.stat(train_ins);
            train_ins.clear();

            // test
            if (true) {
                std::cout << "iter: " << iter << std::endl;
                std::cout << "test:\n";
                auto data_iter_test = dataload(test_file, 1, 100, false);
                for (vector<YCDL::Instance> instances: data_iter_test) {
                    for (auto x: instances) {
                        test_ins.emplace_back(std::move(x));
                    }
                }
                network.stat(test_ins);
                test_ins.clear();
            }
        }
    }
}
