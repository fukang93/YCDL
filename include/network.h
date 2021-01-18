#pragma once

#include "all.h"
#include "instance.h"
#include "optimizer.h"
#include "loss_func.h"
#include "matrix_val.h"
#include "dense_layer.h"
#include "ioc.h"
#include "tools.h"

namespace YCDL {
    class Network {
    public:
        std::shared_ptr<Optimizer> _optimizer;
        std::vector<std::shared_ptr<MatrixValue> > _matrix_vec;
        std::vector<std::shared_ptr<NNLayer>> _layer_vec;
        std::shared_ptr<MatrixValue> _input;
        std::shared_ptr<MatrixValue> _label;

        void load_weight() {
            nlohmann::json &conf = global_conf();

            auto input_params = conf["input_params"];
            std::vector<std::string> slot_vec_str;
            std::vector<int> slot_vec;
            std::string slots_ids = input_params["slots"];
            boost::split(slot_vec_str, slots_ids, boost::is_any_of(","));
            for (auto &str : slot_vec_str) {
                slot_vec.push_back(boost::lexical_cast<int>(str));
            }
            int dim = input_params["dim"];
            bool is_train = true;
            _input = std::make_shared<SparseMatrixValue>();
            _input->init(slot_vec, dim, _optimizer, is_train);
            global_matrix_value().add_matrix_value(input_params["input_name"], _input);
            _label = std::make_shared<DenseMatrixValue>();
            global_matrix_value().add_matrix_value(input_params["label_name"], _label);

            auto params = conf["params"];
            for (int i = 0; i < params.size(); i++) {
                std::string key_str = params[i]["name"];
                ull key = BKDRHash(key_str);
                int row = params[i]["row"];
                int col = params[i]["col"];
                bool need_gradient = params[i]["need_gradient"];
                std::shared_ptr<MatrixValue> m_ptr = std::make_shared<DenseMatrixValue>();
                m_ptr->init(key_str, row, col, _optimizer, need_gradient);
                _matrix_vec.push_back(m_ptr);
                global_matrix_value().add_matrix_value(key_str, m_ptr);
            }
        }

        void load_layer() {
            nlohmann::json &conf = global_conf();
            auto layers = conf["layers"];
            for (int i = 0; i < layers.size(); i++) {
                std::string layer_name = layers[i]["name"];
                std::shared_ptr<NNLayer> tmp_layer = MakeLayer<NNLayer>(layer_name);
                tmp_layer->load_conf(layers[i]);
                _layer_vec.push_back(tmp_layer);
            }
        }

        Network(std::shared_ptr<Optimizer> opt) {
            _optimizer = opt;
            load_weight();
            load_layer();
        }

        void forward(std::vector<Instance> &instances, bool is_train = true) {

            std::vector<std::vector<SLOT_ID_FEAS>> feas_lines;
            std::vector<double> labels;
            for (int j = 0; j < instances.size(); j++) {
                Instance &ins = instances[j];
                feas_lines.push_back(ins.slot_feas);
                labels.push_back(ins.label);
            }
            _input->pull(feas_lines, is_train);
            _label->update_value(labels, labels.size(), 1);

            for (int i = 0; i < _matrix_vec.size(); i++) {
                auto &weight = _matrix_vec[i];
                weight->pull(is_train);
            }

            for (int i = 0; i < _layer_vec.size(); i++) {
                auto &layer = _layer_vec[i];
                layer->forward();
            }
        }

        void backward(std::vector<Instance> &instances) {
            for (int i = _layer_vec.size() - 1; i >= 0; i--) {
                auto &layer = _layer_vec[i];
                layer->backward();
            }

            for (int i = 0; i < _matrix_vec.size(); i++) {
                auto &weight = _matrix_vec[i];
                weight->push();
            }

            std::vector<std::vector<SLOT_ID_FEAS>> feas_lines;
            for (int j = 0; j < instances.size(); j++) {
                Instance &ins = instances[j];
                feas_lines.push_back(ins.slot_feas);
            }
            _input->push(feas_lines);
        }

        void stat(std::vector<Instance> &instances) {

            auto label = global_matrix_value().get_matrix_value("label");
            auto predict = global_matrix_value().get_matrix_value("predict");
            forward(instances, false);

            double loss = 0.0;
            double pre_avg = 0.0;
            double label_avg = 0.0;
            std::vector<std::pair<int, double> > label_pre;
            for (int i = 0; i < label->_val.rows(); i++) {
                for (int j = 0; j < label->_val.cols(); j++) {
                    double val = label->_val(i, j) - predict->_val(i, j);
                    int l = 0;
                    if (label->_val(i, j) > 0.5) l = 1;
                    label_pre.push_back(std::make_pair(l, predict->_val(i, j)));
                    loss += val * val / 2;
                    label_avg += label->_val(i, j);
                    pre_avg += predict->_val(i, j);
                }
            }
            loss /= label->_val.size();
            label_avg /= label->_val.size();
            pre_avg /= label->_val.size();
            std::cout << "loss: " << loss << std::endl;
            std::vector<double> metric = calc(label_pre);
            std::cout << "auc: " << calc_auc(label_pre) << ", loss: " << loss << ", acc: " << metric[0]
                 << ", pre: " << metric[1] << ", recall: " << metric[2] << ", pre_avg: " << pre_avg << ", label_avg: "
                 << label_avg << std::endl;
        }
    };
}
