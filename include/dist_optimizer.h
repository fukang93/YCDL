#pragma once

#include "ps/ps.h"
#include "all.h"
#include "optimizer.h"
#include <mutex>

namespace YCDL {

    // dist worker
    class DistWorker : public Optimizer {
    public:
        DistWorker() {
            _kv = new ps::KVWorker<double>(0, 0);
        }

        virtual void push(std::vector<ull> keys, std::vector<std::vector<double>> &vals) override {
            int n = keys.size();

            // std::cout << "Push请求 " << ps::MyRank() << std::endl;
            // sorted keys
            std::vector<ull> sorted_keys = keys;
            std::sort(sorted_keys.begin(), sorted_keys.end());
            sorted_keys.erase(std::unique(sorted_keys.begin(), sorted_keys.end()), sorted_keys.end());
            int sorted_keys_len = sorted_keys.size();
            std::unordered_map<ull, int> sorted_keys_mp;
            for (int i = 0; i < sorted_keys_len; i++) {
                sorted_keys_mp[sorted_keys[i]] = i;
            }

            // aggregate sorted vals
            std::vector<std::vector<double>> sorted_vals(sorted_keys_len, std::vector<double>());
            std::vector<int> dup_cnt(sorted_keys_len, 0);
            for (int i = 0; i < n; i++) {
                int k = sorted_keys_mp[keys[i]];
                dup_cnt[k]++;
                if (sorted_vals[k].empty()) {
                    sorted_vals[k] = vals[i];
                } else {
                    for (int j = 0; j < vals[i].size(); j++) {
                        sorted_vals[k][j] += vals[i][j];
                    }
                }
            }
            for (int i = 0; i < sorted_keys_len; i++) {
                for (int j = 0; j < sorted_vals[i].size(); j++) {
                    sorted_vals[i][j] /= dup_cnt[i];
                }
            }

            // prepare push values
            std::vector<double> values;
            std::vector<int> lens;
            for (auto &val : sorted_vals) {
                lens.push_back(val.size());
                for (auto x : val) {
                    values.push_back(x);
                }
            }
            _kv->Wait(_kv->Push(sorted_keys, values, lens));
        }

        virtual int pull(std::vector<ull> keys, std::vector<std::vector<double>> &vals, bool is_train = true, int dim = 1) override {

            int n = keys.size();

            // sorted keys
            std::vector<ull> sorted_keys = keys;
            std::sort(sorted_keys.begin(), sorted_keys.end());
            sorted_keys.erase(std::unique(sorted_keys.begin(), sorted_keys.end()), sorted_keys.end());
            int sorted_keys_len = sorted_keys.size();
            std::unordered_map<ull, int> sorted_keys_mp;
            for (int i = 0; i < sorted_keys_len; i++) {
                sorted_keys_mp[sorted_keys[i]] = i;
            }

            std::vector<double> values, values_init;
            std::vector<int> lens;
            for (auto &key : sorted_keys) {
                lens.push_back(dim);
                for (int i = 0; i < dim; i++) {
                    values_init.push_back(0.0);
                }
            }

            if (is_train) {
                // init value
                //std::cout << "Push请求 init " << ps::MyRank() << std::endl;
                _kv->Wait(_kv->Push(sorted_keys, values_init, lens));
            }

            //std::cout << "Pull请求 " << ps::MyRank() << std::endl;
            _kv->Wait(_kv->Pull(sorted_keys, &(values), &(lens)));
            //std::cout << "Pull请求 返回" << std::endl;

            for (int i = 0; i < n; i++) {
                int k = sorted_keys_mp[keys[i]];
                for (int j = 0; j < dim; j++) {
                    vals[i][j] = values[dim * k + j];
                }
            }
            return 0;
        }

        ps::KVWorker<double> *_kv;
    };


    // dist server
    class PushReq {
    public:
        PushReq(const ps::KVMeta &req_meta_tmp, const ps::KVPairs<double> &req_data_tmp) {
            req_meta = req_meta_tmp;
            req_data = req_data_tmp;
        }

        ps::KVMeta req_meta;
        ps::KVPairs<double> req_data;
    };

    class SGDServer : public Layer {
    public:
        SGDServer(int sync_mode = 1, float learning_rate = 0.01) {
            _sync_mode = sync_mode;
            _learning_rate = learning_rate;
            _ps_server = new ps::KVServer<double>(0);
            _ps_server->set_request_handle(
                    std::bind(&SGDServer::DataHandle, this, std::placeholders::_1, std::placeholders::_2,
                              std::placeholders::_3));
        }

        ~SGDServer() {}

        void DataHandle(const ps::KVMeta &req_meta,
                        const ps::KVPairs<double> &req_data,
                        ps::KVServer<double> *server) {

            size_t n = req_data.keys.size();

            if (req_meta.push) {
                //std::cout << ps::MyRank() << " server get push request " << n << " keys\n";

                for (int i = 0; i < n; i++) {
                    auto &key = req_data.keys[i];
                    int length = req_data.lens[i];

                    //init value
                    if (_weights.count(key) == 0) {
                        // use lens[n] represent is_train
                        std::vector<double> w;
                        for (int j = 0; j < length; j++) {
                            w.push_back(get_random_double());
                        }
                        _weights[key] = std::move(w);
                    }
                }

                // deal func
                auto deal_push_req = [&](const ps::KVMeta &req_meta_tmp, const ps::KVPairs<double> &req_data_tmp) {
                    int idx = 0;
                    size_t n = req_data_tmp.keys.size();
                    for (size_t i = 0; i < n; ++i) {
                        auto &key = req_data_tmp.keys[i];
                        auto &weights = _weights[key];
                        int length = req_data_tmp.lens[i];
                        for (int j = 0; j < length; j++) {
                            weights[j] -= _learning_rate * req_data_tmp.vals[idx + j];
                        }
                        idx += length;
                    }
                };
                if (!_sync_mode) {
                    // async push
                    deal_push_req(req_meta, req_data);
                    server->Response(req_meta);
                } else {

                    //std::cout << "server:" << ps::MyRank() << " " <<  req_meta.sender << std::endl;
                    _mutex.lock();
                    if (_push_req.size() == ps::NumWorkers() - 1) {
                        deal_push_req(req_meta, req_data);
                        for (auto &x : _push_req) {
                            deal_push_req(x.req_meta, x.req_data);
                        }
                        server->Response(req_meta);
                        for (auto &x : _push_req) {
                            server->Response(x.req_meta);
                        }
                        _push_req.clear();
                    } else {
                        _push_req.push_back({req_meta, req_data});
                    }
                    _mutex.unlock();
                }
            } else { // pull
                //std::cout << "server get pull request " << n << " keys\n";
                ps::KVPairs<double> response;
                response.keys = req_data.keys;
                int dim = 0;
                for (size_t i = 0; i < n; ++i) {
                    auto &key = req_data.keys[i];
                    std::vector<double> weights(dim, 0.0);
                    if (_weights.count(key)) {
                        weights = _weights[key];
                    }
                    dim = weights.size();
                    response.lens.push_back(weights.size());
                    for (auto x: weights) {
                        response.vals.push_back(x);
                    }
                }
                server->Response(req_meta, response);
                //std::cout << "server send pull response\n";
            }
            //std::cout << "server out :" << ps::MyRank() << " " <<  _push_req.size() << std::endl;
        }

        int _sync_mode;
        float _learning_rate;

        std::vector<PushReq> _push_req;
        std::mutex _mutex;

        std::unordered_map<ps::Key, std::vector<double>> _weights;
        ps::KVServer<double> *_ps_server;

    };


    class AdagradServer : public Layer {
    public:
        explicit AdagradServer(int sync_mode = 1, float learning_rate = 0.1) {
            _sync_mode = sync_mode;
            _learning_rate = learning_rate;
            _ps_server = new ps::KVServer<double>(0);
            _eps = 1e-7;
            _ps_server->set_request_handle(
                    std::bind(&AdagradServer::DataHandle, this, std::placeholders::_1, std::placeholders::_2,
                              std::placeholders::_3));
        }

        void DataHandle(const ps::KVMeta &req_meta,
                        const ps::KVPairs<double> &req_data,
                        ps::KVServer<double> *server) {

            size_t n = req_data.keys.size();

            if (req_meta.push) {
                //std::cout << ps::MyRank() << " server get push request " << n << " keys\n";

                for (int i = 0; i < n; i++) {
                    auto &key = req_data.keys[i];
                    int length = req_data.lens[i];

                    //init value
                    if (_weights.count(key) == 0) {
                        // use lens[n] represent is_train
                        std::vector<double> w, w2;
                        for (int j = 0; j < length; j++) {
                            w.push_back(get_random_double());
                            w2.push_back(0.0);
                        }
                        _weights[key] = std::move(w);
                        _weights2[key] = std::move(w2);
                    }
                }

                // deal func
                auto deal_push_req = [&](const ps::KVMeta &req_meta_tmp, const ps::KVPairs<double> &req_data_tmp) {
                    int idx = 0;
                    size_t n = req_data_tmp.keys.size();
                    for (size_t i = 0; i < n; ++i) {
                        auto &key = req_data_tmp.keys[i];
                        auto &weights = _weights[key];
                        auto &weights2 = _weights2[key];
                        int length = req_data_tmp.lens[i];
                        for (int j = 0; j < length; j++) {
                            weights2[j] += req_data_tmp.vals[idx + j] * req_data_tmp.vals[idx + j];
                            weights[j] -= _learning_rate * req_data_tmp.vals[idx + j] / (_eps + std::sqrt(weights2[j]));
                        }
                        idx += length;
                    }
                };
                if (!_sync_mode) {
                    // async push
                    deal_push_req(req_meta, req_data);
                    server->Response(req_meta);
                } else {

                    //std::cout << "server:" << ps::MyRank() << " " <<  req_meta.sender << std::endl;
                    _mutex.lock();
                    if (_push_req.size() == ps::NumWorkers() - 1) {
                        deal_push_req(req_meta, req_data);
                        for (auto &x : _push_req) {
                            deal_push_req(x.req_meta, x.req_data);
                        }
                        server->Response(req_meta);
                        for (auto &x : _push_req) {
                            server->Response(x.req_meta);
                        }
                        _push_req.clear();
                    } else {
                        _push_req.push_back({req_meta, req_data});
                    }
                    _mutex.unlock();
                }
            } else { // pull
                //std::cout << "server get pull request " << n << " keys\n";
                ps::KVPairs<double> response;
                response.keys = req_data.keys;
                int dim = 0;
                for (size_t i = 0; i < n; ++i) {
                    auto &key = req_data.keys[i];
                    std::vector<double> weights(dim, 0.0);
                    if (_weights.count(key)) {
                        weights = _weights[key];
                    }
                    dim = weights.size();
                    response.lens.push_back(weights.size());
                    for (auto x: weights) {
                        response.vals.push_back(x);
                    }
                }
                server->Response(req_meta, response);
                //std::cout << "server send pull response\n";
            }
            //std::cout << "server out :" << ps::MyRank() << " " <<  _push_req.size() << std::endl;
        }

        double _eps;
        std::unordered_map<ps::Key, std::vector<double>> _weights2;

        int _sync_mode;
        float _learning_rate;

        std::vector<PushReq> _push_req;
        std::mutex _mutex;

        std::unordered_map<ps::Key, std::vector<double>> _weights;
        ps::KVServer<double> *_ps_server;
    };
}
