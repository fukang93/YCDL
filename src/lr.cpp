#include "all.h"
#include "tools.h"
#include "dataload.h"
#include "lr.h"
#include "ioc.h"
#include "init.h"

int main() {
    YCDL::initialize();

    nlohmann::json& conf = YCDL::global_conf();
    std::ifstream ifs("conf/config.json");
    ifs >> conf;

    int epoch = conf["epoch"];
    std::string train_file = conf["train_file"];
    std::string test_file = conf["test_file"];
    int batch_size = conf["batch_size"];
    int shuffle_num = conf["shuffle_num"];
    std::string optimizer_name = conf["optimizer"]["name"];
    std::string loss_func_name = conf["loss_func_name"];
    double learning_rate = conf["optimizer"]["learning_rate"];
    int dim = conf["optimizer"]["dim"];

    std::shared_ptr<YCDL::Optimizer> optimizer = YCDL::MakeLayer<YCDL::Optimizer>(optimizer_name);
    std::shared_ptr<YCDL::loss_func_layer> loss_function = YCDL::MakeLayer<YCDL::loss_func_layer>(loss_func_name);
    YCDL::LRmodel lr = YCDL::LRmodel(optimizer, loss_function, dim);
    
    int total_num = YCDL::get_file_len(train_file);
    int epoch_batch_num = total_num / batch_size;
    int iter = 0;

    auto data_iter = YCDL::dataload(train_file, epoch, batch_size, true, shuffle_num);
    for (std::vector<YCDL::Instance> instances : data_iter) {
        iter++;

        lr.forward(instances);
        lr.backward(instances);
        // test auc
        std::vector<YCDL::Instance> train_ins, test_ins;
        if (iter % epoch_batch_num == 0) {
            int epoch_num = iter / epoch_batch_num;
            // learning_rate adjust

            std::cout << "epoch_num: " << epoch_num << std::endl;
            std::cout << "train:\n";
            auto data_iter_test = YCDL::dataload(train_file, 1, 100, false);
            for (std::vector<YCDL::Instance> instances: data_iter_test) {
                if (train_ins.size() > 100000) break;
                for (auto x: instances) {
                    train_ins.emplace_back(std::move(x));
                }
            }
            lr.stat(train_ins);
            train_ins.clear();
        }
        
        if (iter % 100000 == 0) {
            std::cout << "iter: " << iter << std::endl;
            std::cout << "test:\n";
            auto data_iter_test = YCDL::dataload(test_file, 1, 100, false);
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
