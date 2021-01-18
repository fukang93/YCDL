#include <generator.h>
#include <iostream>

int main() {

    auto fibonacci_numbers = YCDL::Generator<long>([=](YCDL::Yield<long> &yield) {
        long a = 0, b = 1;
        yield(a);
        yield(b);
        while (true) {
            long t = a + b;
            yield(t);
            a = b;
            b = t;
        }
    });

    for (auto i:fibonacci_numbers) {
        std::cout << i << std::endl;
        if (i > 10000) break;
    }

}
