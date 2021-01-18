#include "all.h"
#include "tools.h"
#include "lr.h"
#include "ioc.h"
#include "init.h"
#include "instance.h"
#include <generator.h>
#include "ps/ps.h"
#include "dist_optimizer.h"
#include <algorithm>

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

YCDL::Generator<std::vector< YCDL::Instance> > dataload(std::string path, int epoch, int batch_size, bool is_train, int shuffle_num = 1) {
    return YCDL::Generator<std::vector< YCDL::Instance> >([=](YCDL::Yield<std::vector< YCDL::Instance> > &yield) {
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
                        // cout << delim << " delim\n";
                        value = deal_num(segs[i], delim, heads[i]);
                    }
                    ins.feas.push_back(value);
                    ins.vals.push_back(1.0);
                    //cout << heads[i] << " " << ways[i] <<  " " << i << " " << value  << " " << segs[i] << endl;
                }
                ins.feas.push_back(0);
                ins.vals.push_back(1.0);
                ins.label = boost::lexical_cast<int>(deal_label(segs[len-1]));
                //cout << deal_label(segs[len-1]) << endl;

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

void start_server() {

    auto & conf = YCDL::global_conf();
    std::string optimizer_name = conf["optimizer"]["name"];
    auto server = YCDL::MakeLayer2<YCDL::Layer>(optimizer_name);
    ps::RegisterExitCallback([server]() { delete server; });
}

void start_work() {

    int rank = ps::MyRank();
    auto & conf = YCDL::global_conf();
    int epoch = conf["epoch"];
    std::string train_file = "./data/uci_data/dist_bank/train_" + std::to_string(rank);
    std::string test_file = "./data/uci_data/bank/test";
    int batch_size = conf["batch_size"];
    int shuffle_num = conf["shuffle_num"];
    std::string optimizer_name = "DistWorker";
    std::string loss_func_name = conf["loss_func_name"];
    int dim = conf["optimizer"]["dim"];

    std::shared_ptr<YCDL::Optimizer> optimizer = YCDL::MakeLayer<YCDL::Optimizer>(optimizer_name);
    std::shared_ptr<YCDL::loss_func_layer> loss_function = YCDL::MakeLayer<YCDL::loss_func_layer>(loss_func_name);
    YCDL::LRmodel lr = YCDL::LRmodel(optimizer, loss_function, dim);

    int total_num = YCDL::get_file_len(train_file);
    int epoch_batch_num = total_num / batch_size;
    int iter = 0;

    auto data_iter = dataload(train_file, epoch, batch_size, true, shuffle_num);
    for (std::vector<YCDL::Instance> instances : data_iter) {
        iter++;
        //cout << "print weight\n";
        lr.forward(instances);
        //optimizer->print_weight();
        lr.backward(instances);
        // test auc
        std::vector<YCDL::Instance> train_ins, test_ins;
        int epoch_num = iter / epoch_batch_num;
        if (iter % epoch_batch_num == 0 && rank == 0) {

            std::cout << "epoch_num: " << epoch_num << std::endl;
            std::cout << "train:\n";

            auto data_iter_test = dataload("./data/uci_data/bank/train", 1, 100, false);
            for (std::vector<YCDL::Instance> instances: data_iter_test) {
                if (train_ins.size() > 100000) break;
                for (auto x: instances) {
                    train_ins.emplace_back(std::move(x));
                }
            }
            lr.stat(train_ins);
            train_ins.clear();

            // test
            if (true) {
                std::cout << "iter: " << iter << std::endl;
                std::cout << "test:\n";
                auto data_iter_test = dataload(test_file, 1, 100, false);
                for (std::vector<YCDL::Instance> instances: data_iter_test) {
                    for (auto x: instances) {
                        test_ins.emplace_back(std::move(x));
                    }
                }
                lr.stat(test_ins);
                test_ins.clear();
            }
        }
    }
}

void global_init() {
    YCDL::initialize();

    nlohmann::json &conf = YCDL::global_conf();
    std::ifstream ifs("conf/dist_config.json");
    ifs >> conf;
}

int main() {

    ps::Start(0);

    global_init();
    if (ps::IsServer()) {

        std::cout << "server:" << ps::MyRank() << " running\n";
        start_server();
    } else if (ps::IsWorker()){
        std::cout << "worker:" << ps::MyRank() << " running\n";
        start_work();
    }

    ps::Finalize(0);
}
