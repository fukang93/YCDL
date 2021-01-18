#pragma once

#include "all.h"
#include "optimizer.h"

namespace YCDL {
    class MatrixValue {
    public:
        virtual void pull(bool is_train = true) {}

        virtual void pull(std::vector<std::vector<SLOT_ID_FEAS>> &lines, bool is_train = true) {}

        virtual void push() {}

        virtual void push(std::vector<std::vector<SLOT_ID_FEAS>> &lines) {}

        virtual void push_over(std::vector<double> val) {}

        virtual void
        init(std::string name, int row, int col, std::shared_ptr<Optimizer> opt, bool need_gradient) {}

        virtual void
        init(std::vector<int> slot_vec, int dim, std::shared_ptr<Optimizer> opt, bool need_gradient) {}

        bool update_value(std::vector<double> &vec, int n, int m) {
            if (vec.size() != n * m) {
                return false;
            }
            _val.setZero(n, m);
            for (int i = 0; i < n * m; i++) {
                _val(i / m, i % m) = vec[i];
            }
            return true;
        }

        bool _trainable = false;
        bool _has_grad = false;
        Eigen::MatrixXf _val, _grad;
    };

    class DenseMatrixValue : public MatrixValue {
    public:
        void init(std::string name, int row, int col, std::shared_ptr<Optimizer> opt,
                  bool trainable) override {
            _name = name;
            _key = BKDRHash(name);
            _row = row;
            _col = col;
            _optimizer = opt;
            _trainable = trainable;
        }

        void pull(bool is_train = true) override {
            std::vector<ull> feas{_key};
            std::vector<std::vector<double>> vals(1, std::vector<double>(_row * _col, 0.0));
            _optimizer->pull(feas, vals, is_train, _row * _col);
            _val.setZero(_row, _col);
            _has_grad = false;
            for (int i = 0; i < _row * _col; i++) {
                _val(i / _col, i % _col) = vals[0][i];
            }
        }

        void push() override {
            if (!_trainable || !_has_grad) {
                return;
            }
            std::vector<double> val;
            for (int i = 0; i < _row; i++) {
                for (int j = 0; j < _col; j++) {
                    val.push_back(_grad(i, j));
                }
            }
            std::vector<ull> feas{_key};
            std::vector<std::vector<double>> vals(1, val);
            _optimizer->push(feas, vals);
        }

        void push_over(std::vector<double> val) override {
            std::vector<ull> feas{_key};
            std::vector<std::vector<double>> vals(1, val);
            _optimizer->push_over(feas, vals);
        }

        std::shared_ptr<Optimizer> _optimizer;
        int _row;
        int _col;
        std::string _name;
        ull _key;
    };

    class SparseMatrixValue : public MatrixValue {
    public:
        void
        init(std::vector<int> slot_vec, int dim, std::shared_ptr<Optimizer> opt, bool trainable) override {
            _slot_vec = slot_vec;
            _dim = dim;
            _optimizer = opt;
            _trainable = trainable;
            for (int i = 0; i < slot_vec.size(); i++) {
                _slot_id_to_idx_mp[slot_vec[i]] = i;
            }
        }

        void pull(std::vector<std::vector<SLOT_ID_FEAS>> &lines, bool is_train = true) override {

            int batch_size = lines.size();
            // prepare feas
            std::vector<ull> feas;
            for (auto &line : lines) {
                for (auto &slot_pair : line) {
                    int slot_id = slot_pair.first;
                    std::vector<std::string> &feas_str = slot_pair.second;
                    for (auto fea_str : feas_str) {
                        std::string slot_fea = boost::lexical_cast<std::string>(slot_id) + ":" + fea_str;
                        ull tmp_key = BKDRHash(slot_fea);
                        feas.push_back(tmp_key);
                    }
                }
            }

            // pull feas
            std::vector<std::vector<double>> vals(feas.size(), std::vector<double>(_dim, 0.0));
            _optimizer->pull(feas, vals, is_train, _dim);

            // matrix value init
            int col = _dim * _slot_vec.size();
            _val.setZero(batch_size, col);
            _has_grad = false;

            // transform feas to matrix value
            int line_id = 0;
            int feas_idx = 0;
            for (auto &line : lines) {
                for (auto &slot_pair : line) {
                    int slot_id = slot_pair.first;
                    std::vector<std::string> &feas_str = slot_pair.second;

                    if (!_slot_id_to_idx_mp.count(slot_id)) {
                        std::cerr << "ERROR slot_id: " << slot_id << std::endl;
                        return;
                    }
                    int slot_idx = _slot_id_to_idx_mp[slot_id];

                    for (auto fea_str : feas_str) {
                        std::vector<double> &val = vals[feas_idx];
                        feas_idx++;
                        if (val.size() != _dim) {
                            std::cerr << "ERROR fea size: " << val.size() << std::endl;
                            return;
                        }
                        for (int i = slot_idx * _dim; i < (slot_idx + 1) * _dim; i++) {
                            _val(line_id, i) += val[i - slot_idx * _dim];
                        }
                    }
                }
                line_id++;
            }
        }

        void push(std::vector<std::vector<SLOT_ID_FEAS>> &lines) override {
            if (!_trainable || !_has_grad) {
                return;
            }

            std::vector<ull> feas;
            std::vector<std::vector<double>> vals;

            int line_id = 0;
            for (auto &line : lines) {
                for (auto &slot_pair : line) {
                    int slot_id = slot_pair.first;
                    std::vector<std::string> &feas_str = slot_pair.second;

                    int slot_idx = _slot_id_to_idx_mp[slot_id];

                    for (auto fea_str : feas_str) {
                        std::string slot_fea = boost::lexical_cast<std::string>(slot_id) + ":" + fea_str;
                        ull tmp_key = BKDRHash(slot_fea);
                        feas.push_back(tmp_key);
                        std::vector<double> val;
                        for (int i = slot_idx * _dim; i < (slot_idx + 1) * _dim; i++) {
                            val.push_back(_grad(line_id, i));
                        }
                        vals.emplace_back(val);
                    }
                }
                line_id++;
            }

            _optimizer->push(feas, vals);
        }

        std::vector<int> _slot_vec;
        int _dim;
        std::shared_ptr<Optimizer> _optimizer;
        std::unordered_map<int, int> _slot_id_to_idx_mp;
    };

    class MatrixValueMap {
    public:
        std::shared_ptr<MatrixValue> get_matrix_value(std::string name) {
            return _matrix_value_map[name];
        }

        void add_matrix_value(std::string name, std::shared_ptr<MatrixValue> ma) {
            _matrix_value_map[name] = ma;
        }

        std::unordered_map<std::string, std::shared_ptr<MatrixValue>> _matrix_value_map;
    };

    inline MatrixValueMap &global_matrix_value() {
        static MatrixValueMap mp;
        return mp;
    }
}
