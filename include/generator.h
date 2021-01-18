#pragma once
#include <thread>
#include <future>
#include <memory>
#include <exception>
#include <iterator>

namespace YCDL {

    template<class T>
    class Generator;

    template<class T>
    class Yield {

        enum State {
            WAITING, READY, TERMINATED, FINISHED
        } state = WAITING;

        friend Generator<T>;
        std::promise<T> promise;
        std::mutex mutex;
        std::condition_variable ready_wait;

        struct FinishedException : public std::exception {
            const char *what() const throw() override { return "FinishedException"; }
        };

        struct TerminatedException : public std::exception {
            const char *what() const throw() override { return "TerminatedException"; }
        };

        void terminate() {
            std::lock_guard<std::mutex> guard(mutex);
            if (state != FINISHED) {
                state = TERMINATED;
                ready_wait.notify_one();
            }
        }

        void finish() {
            std::unique_lock<std::mutex> lock(mutex);
            while (state == WAITING) ready_wait.wait(lock);
            if (state == TERMINATED) return;
            state = FINISHED;
            throw FinishedException();
        }

        template<class E>
        void set_exception(E e) {
            std::unique_lock<std::mutex> lock(mutex);
            while (state == WAITING) ready_wait.wait(lock);
            if (state == TERMINATED) return;
            promise.set_exception(e);
        }

        std::future<T> get_future() {
            std::lock_guard<std::mutex> guard(mutex);
            promise = std::promise<T>();
            ready_wait.notify_one();
            state = READY;
            return promise.get_future();
        }

        Yield() {}

    public:

        void operator()(const T &value) {
            std::unique_lock<std::mutex> lock(mutex);
            while (state == WAITING) ready_wait.wait(lock);
            if (state == TERMINATED) throw TerminatedException();
            promise.set_value(value);
            state = WAITING;
        }

    };

    template<class T>
    class Generator {
        std::function<void(Yield<T> &)> generator_function;

    public:
        Generator(const std::function<void(Yield<T> &)> &gf) : generator_function(gf) {}

        class const_iterator : public std::iterator<std::input_iterator_tag, T> {
            friend Generator;

            struct Data {
                Yield<T> yield;
                T current_value;
                std::thread thread;

                ~Data() {
                    yield.terminate();
                    thread.join();
                }
            };

            std::unique_ptr<Data> data;

        public:

            T &operator*() const {
                return data->current_value;
            }

            T *operator->() const {
                return &data->current_value;
            }

            const_iterator &operator++() {
                try {
                    data->current_value = data->yield.get_future().get();
                }
                catch (typename Yield<T>::FinishedException) {
                    data.reset();
                }
                return *this;
            }

            bool operator!=(const const_iterator &other) const { return data != other.data; }
        };

        const_iterator begin() const {
            const_iterator it;
            it.data.reset(new typename const_iterator::Data);
            auto data = it.data.get();

            data->thread = std::thread([this, data]() {
                try {
                    generator_function(data->yield);
                    data->yield.finish();
                }
                catch (typename Yield<T>::TerminatedException) {

                }
                catch (...) {
                    data->yield.set_exception(std::current_exception());
                }
            });

            ++it;

            return it;
        }

        const_iterator end() const {
            return const_iterator();
        }
    };
}

