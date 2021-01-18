#pragma once
#include "all.h"
#include "instance.h"
#include "optimizer.h"
#include "loss_func.h"

namespace YCDL {
    class LRmodel {
    public:
        std::shared_ptr<Optimizer> _sparse_weight_sgd;
        std::shared_ptr<loss_func_layer> _loss_layer;
        int _dim;

        LRmodel(std::shared_ptr<Optimizer> &sparse_weight_sgd, std::shared_ptr<loss_func_layer> &loss_layer,
                int len = 1) {
            _dim = len;
            _sparse_weight_sgd = sparse_weight_sgd;
            _loss_layer = loss_layer;
        }

        void forward(std::vector<Instance> &instances, bool is_train = true) {
            for (int j = 0; j < instances.size(); j++) {
                double logit = 0.0;
                Instance &ins = instances[j];
                std::vector<std::vector<double>> weight(ins.feas.size(), std::vector<double>(_dim, 0.0));
                _sparse_weight_sgd->pull(ins.feas, weight, is_train, _dim);
                for (int k = 0; k < ins.feas.size(); k++) {
                    for (int l = 0; l < _dim; l++) {
                        logit += weight[k][l] * ins.vals[k];
                    }
                }
                ins.pre = logit;
            }
        }

        void backward(std::vector<Instance> &instances) {
            for (int j = 0; j < instances.size(); j++) {
                Instance &ins = instances[j];
                double grad_label = _loss_layer->backward(ins.pre, ins.label);
                std::vector<std::vector<double>> grads(ins.feas.size(), std::vector<double>(_dim, 0.0));
                for (int k = 0; k < ins.feas.size(); k++) {
                    for (int l = 0; l < _dim; l++) {
                        grads[k][l] = grad_label * ins.vals[k];
                    }
                }
                _sparse_weight_sgd->push(ins.feas, grads);
            }
        }

        void stat(std::vector<Instance> &instances) {
            // stat
            double loss = 0.0;
            double pre_avg = 0.0;
            double label_avg = 0.0;
            std::vector<std::pair<int, double> > label_pre;
            forward(instances, false);
            for (int j = 0; j < instances.size(); j++) {
                Instance &ins = instances[j];
                double logit = ins.pre;
                double predict = sigmoid(logit);
                label_pre.push_back(std::make_pair(ins.label, predict));
                loss += _loss_layer->forward(logit, ins.label);
                pre_avg += predict;
                label_avg += ins.label;
            }
            pre_avg /= instances.size();
            label_avg /= instances.size();
            loss /= instances.size();
            std::vector<double> out = calc(label_pre, pre_avg);
            std::cout << "auc: " << calc_auc(label_pre) << ", loss: " << loss << ", acc: " << out[0]
                 << ", pre: " << out[1] << ", recall: " << out[2] << ", pre_avg: " << pre_avg << ", label_avg: "
                 << label_avg << std::endl;
        }
    };
}
