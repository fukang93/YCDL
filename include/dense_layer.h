#pragma once
#include "all.h"
#include "matrix_val.h"
#include "eigen_func.h"

namespace YCDL {
    class NNLayer : public Layer {
    public:
        virtual void load_conf(const nlohmann::json &conf) = 0;

        virtual void forward() = 0;

        virtual void backward() = 0;
    };

    class Concat: public NNLayer {
    public:
        void load_conf(const nlohmann::json &conf) override {
            auto inputs = conf["input"];
            for (int i = 0; i < inputs.size(); i++) {
                auto name = inputs[i];
                _input_vec.push_back(global_matrix_value().get_matrix_value(name));
            }

            _output = std::make_shared<DenseMatrixValue>();
            std::string out_name = conf["output"]["name"];
            global_matrix_value().add_matrix_value(out_name, _output);
        }

        void forward() override {
            int input_size = _input_vec.size();
            _output->_has_grad = false;
            _output->_trainable = false;
            int row = 0, col = 0;
            for (int i = 0; i < input_size; i++) {
                row = _input_vec[i]->_val.rows();
                col += _input_vec[i]->_val.cols();
                if (_input_vec[i]->_trainable) {
                    _output->_trainable = true;
                }
            }

            Eigen::MatrixXf &out_val = _output->_val;
            out_val.setZero(row, col);

            int idx = 0;
            for (int i = 0; i < input_size; i++) {
                auto &input_val = _input_vec[i]->_val;
                int input_col = input_val.cols();
                for (int j = 0; j < row; j++) {
                    for (int k = 0; k < input_col; k++) {
                        out_val(j, idx + k) = input_val(j, k);
                    }
                }
                idx += input_col;
            }
        }

        void backward() override {
            if (!_output->_has_grad) {
                return;
            }
            int input_size = _input_vec.size();
            Eigen::MatrixXf &out_grad = _output->_grad;

            int row = out_grad.rows();
            int col = out_grad.cols();

            int idx = 0;
            for (int i = 0; i < input_size; i++) {
                if (!_input_vec[i]->_trainable) {
                    continue;
                }
                int input_col = _input_vec[i]->_val.cols();
                auto &input_grad = _input_vec[i]->_grad;
                if (!_input_vec[i]->_has_grad) {
                    _input_vec[i]->_grad.resize(row, input_col);
                }
                for (int j = 0; j < row; j++) {
                    for (int k = 0; k < input_col; k++) {
                        if (_input_vec[i]->_has_grad) {
                            input_grad(j, k) += out_grad(j, idx + k);
                        } else {
                            input_grad(j, k) = out_grad(j, idx + k);
                        }
                    }
                }
                idx += input_col;
            }
        }

        std::vector <std::shared_ptr<MatrixValue>> _input_vec;
        std::shared_ptr <MatrixValue> _output;
    };

    class Activation : public NNLayer {
    public:
        void load_conf(const nlohmann::json &conf) override {
            _input = global_matrix_value().get_matrix_value(conf["input"]);

            _output = std::make_shared<DenseMatrixValue>();
            std::string out_name = conf["output"]["name"];
            global_matrix_value().add_matrix_value(out_name, _output);

            _act = conf["act"];

        }

        void forward() override {
            Eigen::MatrixXf val = _input->_val;
            Eigen::MatrixXf &out_val = _output->_val;
            if (_act == "sigmoid") {
                // cout << "sigmoid\n";
                out_val = 1.0 / (1.0 + (-val.array()).exp());
                //out_val = val;
            }
            _output->_has_grad = false;
            _output->_trainable = _input->_trainable;
        }

        void backward() override {
            if (!_output->_has_grad) {
                return;
            }
            Eigen::MatrixXf &val = _input->_val;
            Eigen::MatrixXf out_val = _output->_val;
            Eigen::MatrixXf &out_grad = _output->_grad;

            if (_input->_trainable) {
                Eigen::MatrixXf &grad = _input->_grad;
                Eigen::MatrixXf val2;
                if (_act == "sigmoid") {
                    //val2 = out_grad;

                    Eigen::MatrixXf ones;
                    ones.setOnes(out_val.rows(), out_val.cols());
                    val2 = out_val.cwiseProduct(ones - out_val);
                    val2 = val2.cwiseProduct(out_grad);
                }
                if (_input->_has_grad) {
                    grad += val2;
                } else {
                    grad = val2;
                }
                _input->_has_grad = true;
            }
        }

        std::shared_ptr <MatrixValue> _input;
        std::shared_ptr <MatrixValue> _output = std::make_shared<DenseMatrixValue>();
        std::string _act;
    };


    class Dense : public NNLayer {
    public:

        void load_conf(const nlohmann::json &conf) override {
            _input1 = global_matrix_value().get_matrix_value(conf["input1"]);
            _input2 = global_matrix_value().get_matrix_value(conf["input2"]);

            _output = std::make_shared<DenseMatrixValue>();
            std::string out_name = conf["output"]["name"];
            global_matrix_value().add_matrix_value(out_name, _output);

            if (conf.find("trans1") != conf.end()) {
                _trans1 = conf["trans1"];
            } else {
                _trans1 = false;
            }
            if (conf.find("trans2") != conf.end()) {
                _trans2 = conf["trans2"];
            } else {
                _trans2 = false;
            }
            if (conf.find("has_bias") != conf.end()) {
                _has_bias = conf["has_bias"];
            } else {
                _has_bias = false;
            }

            if (_has_bias) {
                _concat_layer = std::make_shared<Concat>();
                _input_bias = std::make_shared<MatrixValue>();
                _concat_layer->_output = std::make_shared<DenseMatrixValue>();
                std::string out_name_bias = out_name + "_bias";
                global_matrix_value().add_matrix_value(out_name_bias, _concat_layer->_output);
            }
        }

        void forward() override {
            Eigen::MatrixXf &val1 = _input1->_val;

            if (_has_bias) {
                auto &bias_val = _input_bias->_val;
                int row = _input1->_val.rows();
                bias_val.setOnes(row, 1);
                _concat_layer->_input_vec.clear();
                _concat_layer->_input_vec.push_back(_input1);
                _concat_layer->_input_vec.push_back(_input_bias);

                _concat_layer->forward();
                val1 = _concat_layer->_output->_val;
            }

            Eigen::MatrixXf &val2 = _input2->_val;
            Eigen::MatrixXf &out_val = _output->_val;

            matrix_multi(_trans1, _trans2, false, val1, val2, out_val);
            _output->_has_grad = false;
            if (_input1->_trainable || _input2->_trainable) {
                _output->_trainable = true;
            } else {
                _output->_trainable = false;
            }
        }

        void backward() override {
            if (!_output->_has_grad) {
                return;
            }

            auto input1_tmp = _input1;

            if (_has_bias) {
                input1_tmp = _concat_layer->_output;
            }

            Eigen::MatrixXf &val1 = input1_tmp->_val;
            Eigen::MatrixXf &val2 = _input2->_val;
            Eigen::MatrixXf &out_grad = _output->_grad;

            if (input1_tmp->_trainable) {
                Eigen::MatrixXf &grad1 = input1_tmp->_grad;
                if (input1_tmp->_has_grad) {
                    matrix_multi_addition(false, !_trans2, _trans1, out_grad, val2, grad1);
                } else {
                    matrix_multi(false, !_trans2, _trans1, out_grad, val2, grad1);
                }
                input1_tmp->_has_grad = true;
            }

            if (_input2->_trainable) {
                Eigen::MatrixXf &grad2 = _input2->_grad;
                if (_input2->_has_grad) {
                    matrix_multi_addition(!_trans1, false, _trans2, val1, out_grad, grad2);
                } else {
                    matrix_multi(!_trans1, false, _trans2, val1, out_grad, grad2);
                }
                _input2->_has_grad = true;
            }

            if (_has_bias) {
                _concat_layer->backward();
            }
        }

        std::shared_ptr <MatrixValue> _input1;
        std::shared_ptr <MatrixValue> _input2;
        std::shared_ptr <MatrixValue> _output;
        bool _trans1 = false;
        bool _trans2 = false;
        bool _has_bias = false;
        std::shared_ptr <Concat> _concat_layer;
        std::shared_ptr <MatrixValue> _input_bias;
    };

    class Loss : public NNLayer {
    public:
        void load_conf(const nlohmann::json &conf) override {
            _label = global_matrix_value().get_matrix_value(conf["label"]);
            _input = global_matrix_value().get_matrix_value(conf["input"]);
            _loss_func = conf["loss"];

            // for auc stat
            global_matrix_value().add_matrix_value("predict", _input);
        }

        void forward() override {
        }

        void backward() override {
            if (!_input->_trainable) {
                return;
            }
            if (!_input->_has_grad) {
                _input->_grad.setZero(_input->_val.rows(), _input->_val.cols());
                _input->_has_grad = true;
            }
            if (_loss_func == "logloss") {
                auto tmp = _input->_val;
                tmp = 1.0 / (1.0 + (-_input->_val.array()).exp());
                _input->_grad += tmp - _label->_val;
            } else if (_loss_func == "mse") {
                _input->_grad += _input->_val - _label->_val;
            }
        }

        std::shared_ptr <MatrixValue> _label;
        std::shared_ptr <MatrixValue> _input;
        std::string _loss_func;
    };
}
