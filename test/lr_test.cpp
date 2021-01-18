#include "all.h"
#include "tools.h"
#include "dataload.h"
#include "lr.h"
#include "optimizer.h"
#include "ioc.h"
#include "init.h"
#include <cstdlib>
#include <ctime>

YCDL::Generator<std::vector<YCDL::Instance> > dataload(std::vector<YCDL::Instance>& input, int epoch, int batch_size, bool is_train) {
    return YCDL::Generator<std::vector<YCDL::Instance> >([=](YCDL::Yield<std::vector<YCDL::Instance> > &yield) {
            int cnt = 0;
        std::vector<YCDL::Instance> instances;
            int nums = epoch;
            while(nums) {
                nums--;
                for (auto x : input) {
                    cnt += 1;
                    instances.emplace_back(std::move(x));
                    if (cnt == batch_size) {
                        cnt = 0;
                        if (is_train) {
                            std::random_shuffle(instances.begin(), instances.end());
                        }
                        yield(instances);
                        instances.clear();
                    }
                }
            }
            if (cnt != batch_size && instances.size() != 0)
                if (is_train) {
                    std::random_shuffle(instances.begin(), instances.end());
                }
                yield(instances);
                instances.clear();
            });
}

int main() {
    YCDL::initialize();

    nlohmann::json& conf = YCDL::global_conf();
    std::ifstream ifs("conf/config.json");
    ifs >> conf;


    std::string optimizer_name = conf["optimizer"]["name"];
    std::string loss_func_name = conf["loss_func_name"];
    int dim = conf["optimizer"]["dim"];

    std::shared_ptr<YCDL::Optimizer> optimizer = YCDL::MakeLayer<YCDL::Optimizer>(optimizer_name);
    std::shared_ptr<YCDL::loss_func_layer> loss_function = YCDL::MakeLayer<YCDL::loss_func_layer>(loss_func_name);
    YCDL::LRmodel lr = YCDL::LRmodel(optimizer, loss_function, dim);

    std::vector<YCDL::Instance> train_data, test_data;
    srand((int)time(0));
    for (int i = 0; i < 10000; i++) {
        double x =(rand() % 10) / 10.0;
        double y = x * 2 + (rand()%10 - 4) / 10.0 - 1;
        double bias = 1.0;
        YCDL::Instance ins;
        if (y - x*2 + 1> 0){
            ins.label = 1;
        } else {
            ins.label = 0;
        }
        ins.feas.push_back(0);
        ins.feas.push_back(1);
        ins.feas.push_back(2);
        ins.vals.push_back(1.0);
        ins.vals.push_back(x);
        ins.vals.push_back(y);
        train_data.emplace_back(std::move(ins));
    }

    for (int i = 0; i < 1000; i++) {
        double x =(rand() % 10) / 10.0;
        double y = x * 2 + (rand()%10 - 4) / 10.0  - 1;
        double bias = 1.0;
        YCDL::Instance ins;
        if (y - x*2 + 1> 0) {
            ins.label = 1;
        } else {
            ins.label = 0;
        }
        ins.feas.push_back(0);
        ins.feas.push_back(1);
        ins.feas.push_back(2);
        ins.vals.push_back(1.0);
        ins.vals.push_back(x);
        ins.vals.push_back(y);
        test_data.emplace_back(std::move(ins));
    }
    
    int epoch = 10;
    int batch_size = 50;
    int total_num = 10000;
    int epoch_batch = total_num / batch_size;
    auto train_data_iter = dataload(train_data, epoch, batch_size, true);
    int iter = 0;
    for (std::vector<YCDL::Instance> instances : train_data_iter) {
        lr.forward(instances);
        lr.backward(instances);
        iter += 1;
        if (iter % epoch_batch == 0) {
            int epoch_num = iter / epoch_batch;
            std::cout << "epoch_num: " << epoch_num << std::endl;
            optimizer->print_weight();
            lr.stat(test_data);
        }
    }
}
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
