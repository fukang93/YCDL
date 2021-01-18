#pragma once
#include "all.h"
#include "tools.h"

namespace YCDL {

    class Optimizer : Layer {
    public:
        virtual void push(std::vector<ull> feas, std::vector<std::vector<double>> &vals) {};

        virtual int pull(std::vector<ull> feas, std::vector<std::vector<double>> &vals, bool is_train = true, int dim = 1) {};

        virtual void print_weight() {};

        virtual void init(ull fea, int dim) {};

        virtual void load_conf(const nlohmann::json &conf) {};

        virtual void push_over(std::vector<ull> feas, std::vector<std::vector<double>> &vals) {};
    };

    class SGD : public Optimizer {
    public:
        SGD() {
            load_conf(global_conf()["optimizer"]);
        }

        virtual void push(std::vector<ull> feas, std::vector<std::vector<double>> &vals) override {
            for (int i = 0; i < feas.size(); i++) {
                ull fea = feas[i];
                auto &arr = _mp[fea];
                for (int j = 0; j < vals[i].size(); j++) {
                    arr[j] -= _learning_rate * vals[i][j];
                }
            }
        }

        virtual void push_over(std::vector<ull> feas, std::vector<std::vector<double>> &vals) override {
            for (int i = 0; i < feas.size(); i++) {
                ull fea = feas[i];
                auto &arr = _mp[fea];
                for (int j = 0; j < vals[i].size(); j++) {
                    arr[j] = vals[i][j];
                }
            }
        }

        virtual void init(ull fea, int dim) override {
            std::vector<double> w;
            for (int i = 0; i < dim; i++) {
                w.push_back(get_random_double());
            }
            _mp[fea] = move(w);
        }

        virtual int pull(std::vector<ull> feas, std::vector<std::vector<double>> &vals, bool is_train = true, int dim = 1) override {
            int not_hit = 0;
            for (int i = 0; i < feas.size(); i++) {
                ull fea = feas[i];
                auto iter = _mp.find(fea);
                if (!is_train && iter == _mp.end()) {
                    not_hit++;
                    continue;
                }
                if (iter == _mp.end()) {
                    init(fea, dim);
                }
                // deep cooy
                vals[i] = _mp[fea];
            }
            return not_hit;
        }

        virtual void print_weight() override {
            for (auto x : _mp) {
                std::cout << x.first;
                for (auto y : x.second) {
                    std::cout << " " << y;
                }
                puts("");
            }
        }

        virtual void load_conf(const nlohmann::json &conf) override {
            if (conf.find("learning_rate") != conf.end()) {
                _learning_rate = conf["learning_rate"];
            } else {
                _learning_rate = 0.1;
            }
        }

        double _learning_rate;
        std::unordered_map<ull, std::vector<double> > _mp;
    };

    class Adagrad : public SGD {
    public:

        Adagrad() {
            load_conf(global_conf()["optimizer"]);
        }

        virtual void push(std::vector<ull> feas, std::vector<std::vector<double>> &vals) override {
            for (int i = 0; i < feas.size(); i++) {
                ull fea = feas[i];
                auto &arr = _mp[fea], &arr2 = _mp2[fea];
                for (int j = 0; j < vals[i].size(); j++) {
                    arr2[j] += vals[i][j] * vals[i][j];
                    arr[j] -= _learning_rate * vals[i][j] / (sqrt(arr2[j]) + _eps);
                }
            }
        }

        virtual void init(ull fea, int dim) override {
            std::vector<double> w, w2;
            for (int i = 0; i < dim; i++) {
                w.push_back(get_random_double());
                w2.push_back(0.0);
            }
            _mp[fea] = move(w);
            _mp2[fea] = move(w2);
        }

        virtual void load_conf(const nlohmann::json &conf) override {
            SGD::load_conf(conf);
            if (conf.find("eps") != conf.end()) {
                _eps = conf["eps"];
            } else {
                _eps = 1e-7;
            }
        }

        double _eps;
        std::unordered_map<ull, std::vector<double> > _mp2;
    };
}
