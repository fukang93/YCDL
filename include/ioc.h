#pragma once

#include "all.h"

namespace YCDL {
    template<class T>
    class IocContainer {
    public:
        using FuncType = std::function<std::shared_ptr<T>()>;
        using FuncType2 = std::function<T*()>;


        //注册一个key对应的类型
        template<class ClassType>
        void registerType(std::string key) {
            FuncType func = [] { return std::make_shared<ClassType>(); };
            registerType(key, func);
        }

        std::shared_ptr<T> resolveShared(std::string key) {
            if (m_map.find(key) == m_map.end()) {
                return nullptr;
            }
            auto func = m_map[key];
            return std::shared_ptr<T>(func());
        }

        //注册一个key对应的类型
        template<class ClassType>
        void registerType2(std::string key) {
            FuncType2 func = [] { return new ClassType; };
            registerType2(key, func);
        }

        T* resolveShared2(std::string key) {
            if (m_map2.find(key) == m_map2.end()) {
                return nullptr;
            }
            auto func = m_map2[key];
            return func();
        }

    private:
        void registerType(std::string key, FuncType type) {
            if (m_map.find(key) != m_map.end()) {
                throw std::invalid_argument("this key has exist");
            }
            m_map.emplace(key, type);
        }
        void registerType2(std::string key, FuncType2 type) {
            if (m_map2.find(key) != m_map2.end()) {
                throw std::invalid_argument("this key has exist");
            }
            m_map2.emplace(key, type);
        }

    private:
        std::map<std::string, FuncType> m_map;
        std::map<std::string, FuncType2> m_map2;

    };

    template<class T>
    inline IocContainer<T> &global_layer_factory() {
        static IocContainer<T> f;
        return f;
    }

    template<class T>
    inline std::shared_ptr<T> MakeLayer(std::string name) {
        return global_layer_factory<T>().resolveShared(name);
    }
    template<class T>
    inline T* MakeLayer2(std::string name) {
        return global_layer_factory<T>().resolveShared2(name);
    }

    template<class T1, class T2>
    inline void RegisterLayer(std::string name) {
        global_layer_factory<T1>().template registerType<T2>(name);
    }
    template<class T1, class T2>
    inline void RegisterLayer2(std::string name) {
        global_layer_factory<T1>().template registerType2<T2>(name);
    }

    struct ICar {
        virtual ~ICar() {}

        virtual void test() const = 0;
    };

    struct Car : ICar {
        void test() const {
            std::cout << "Car test" << std::endl;
        }
    };

    struct Bus : ICar {
        void test() const {
            std::cout << "Bus test" << std::endl;
        }
    };

    void test() {
        IocContainer<ICar> ioc;
        ioc.registerType<Bus>("bus");
        ioc.registerType<Car>("car");
        std::shared_ptr<ICar> bus = ioc.resolveShared("bus");
        bus->test();
        return;
    }
}
