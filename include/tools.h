#pragma once
#include "all.h"

namespace YCDL {
    template<class T>
    void print_vec(const std::string &comment, const std::vector <T> segs) {
        std::cout << comment << ": ";
        for (int i = 0; i < segs.size(); i++)
            std::cout << segs[i] << " ";
        puts("");
    }

    template<class T>
    void line_split(const std::string &line, const std::string &sep, std::vector <T> &ans) {
        std::vector <std::string> segs;
        boost::split(segs, line, boost::is_any_of(sep));
        for (int i = 0; i < segs.size(); i++) {
            ans.emplace_back(boost::lexical_cast<T>(segs[i].c_str()));
        }
        return;
    }

    struct RandomDouble {
        RandomDouble(double l = -1, double u = 1) {
            lower_bound = l;
            upper_bound = u;
            unif = std::uniform_real_distribution<double>(lower_bound, upper_bound);
        }

        double get_random() {
            return unif(re);
        }

        double lower_bound;
        double upper_bound;
        std::uniform_real_distribution<double> unif;
        std::default_random_engine re;
    };


    inline double get_random_double() {
        static RandomDouble rd;
        return rd.get_random();
    }

    int get_file_len(std::string path) {
        std::ifstream file;
        file.open(path);
        int cnt = 0;
        std::string s;
        while (getline(file, s)) {
            cnt += 1;
        }
        file.close();
        return cnt;
    }

    double sigmoid(double x) {
        static double overflow = 20.0;
        if (x > overflow) x = overflow;
        if (x < -overflow) x = -overflow;
        return 1.0 / (1.0 + exp(-x));
    }

    bool cmp(std::pair<int, double> a, std::pair<int, double> b) {
        return a.second < b.second;
    }

    double calc_auc(std::vector <std::pair<int, double>> label_pre) {
        sort(label_pre.begin(), label_pre.end(), cmp);
        std::unordered_map<double, std::vector<int> > val_mp;
        int p = 0, n = 0, sum_rank = 0;
        for (int i = 0; i < label_pre.size(); i++) {
            if (label_pre[i].first == 1)
                p++, sum_rank += (i + 1);
            else
                n++;
        }
        int N = n * p;
        if (N == 0)return 0.0;
        return (sum_rank - p * (p + 1) / 2.0) / 1.0 / N;
    }

/*
double calc_auc(vector<pair<int, double> > & label_pre) {
    const int bucket_size = 10000000;
    vector<double> bucket1(bucket_size, 0.0), bucket2(bucket_size, 0.0);
    for (int i = 0; i < label_pre.size(); i++) {
        pair<int, double> p = label_pre[i];
        int bucket_id = min(bucket_size - 1, int(p.second * bucket_size));
        bucket1[bucket_id] += p.second;
        bucket2[bucket_id] += (1 - p.second);
    }
    double tp = 0.0, fp = 0.0, area = 0.0;
    for (int i = bucket2.size() - 1; i >= 0; i--) {
        double newtp = tp + bucket1[i];
        double newfp = fp + bucket2[i];
        area += (newtp + tp) * (newfp - fp) / 2;
        tp = newtp;
        fp = newfp;
    }
    return area / tp / fp;
}
*/

    std::vector<double> calc(std::vector <std::pair<int, double>> &label_pre, double threshord = 0.5) {
        double tp = 0.0, fp = 0.0, tn = 0.0, fn = 0.0;
        for (int i = 0; i < label_pre.size(); i++) {
            std::pair<int, double> &p = label_pre[i];
            // cout << p.first << " " << p.second << endl;
            if (p.first == 1 && p.second > threshord) {
                tp++;
            }
            if (p.first == 0 && p.second < threshord) {
                tn++;
            }
            if (p.first == 0 && p.second > threshord) {
                fp++;
            }
            if (p.first == 1 && p.second < threshord) {
                fn++;
            }
        }

        double acc = (tp + tn) / (tp + tn + fp + fn);
        double pre = 0.0, recall = 0.0;
        if (tp + fp)pre = tp / (tp + fp);
        if (tp + fn)recall = tp / (tp + fn);
        std::vector<double> ans = {acc, pre, recall};
        return ans;
    }
}