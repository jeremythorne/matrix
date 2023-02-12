#include "matrix.h"

void test_mul() {
    Mat A({2, 2}, 
            {1.0, 0.0,
             0.0, 1.0});
    Mat B({2, 2}, 
            {2.0, 0.0,
             0.0, 1.0});
    auto C = mul(A, B);

    assert(eq(C, B));
    assert(A * B == C);

    Mat D({2, 1}, {2.0, 
                   3.0});
    assert(eq(mul(B, D), Mat({2, 1}, {4.0,
                                      3.0})));
}

void test_transpose() {
    Mat A({2, 2}, 
            {2.0, 0.0,
             0.0, 1.0});
    Mat B({3, 2},
            {2.0, -1.0,
             3.0, 1.0,
             1.0, 0.5});
    Mat C({2, 3},
            {2.0, 3.0, 1.0,
             -1.0, 1.0, 0.5});
   
    assert(eq(transpose(A), A));
    assert(eq(transpose(B), C));
    assert(B.t() == C);
    assert(B.t().t() == B);
}

int main() {
    test_mul();
    test_transpose();
    return 0;
}
