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

    auto I = identity<double>(10);
    assert(I * I == I);
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
    auto I = identity<double>(10);
    assert(I.t() == I);
}

void test_add() {
    Mat A({2, 2}, 
            {1.0, 0.0,
             0.0, 1.0});
    Mat B({2, 2}, 
            {2.0, 0.0,
             1.0, 1.0});
    Mat C({2, 2}, 
            {3.0, 0.0,
             1.0, 2.0});
    assert(A + B == C);    
}

void test_sub() {
    Mat A({2, 2}, 
            {1.0, 0.0,
             0.0, 1.0});
    Mat B({2, 2}, 
            {2.0, 0.0,
             1.0, 1.0});
    Mat C({2, 2}, 
            {-1.0, 0.0,
             -1.0, 0.0});
    assert(A - B == C);    
}

void test_LU() {
    Mat A({3, 3},{
     2, -1, -2,
    -4,  6,  3,
    -4, -2,  8});

    Mat Le({3, 3},{
     1,  0,  0,
    -2,  1,  0,
    -2, -1,  1});

    Mat Ue({3, 3},{
     2, -1, -2,
     0,  4, -1,
     0,  0,  3});

    auto [L, U] = LU_decompose(A);

    assert(L == Le);
    assert(U == Ue);
    assert(is_lower_triangular(L));

    Mat ye({3, 1}, {1.0,
                   2.0,
                  -4.0});

    Mat b = L * ye;

    Mat y = solve_Ly_b(L, b);
    assert(is_vector(y));
    assert(y == ye);

    Mat xe({3, 1}, {1.0,
                   2.0,
                  -4.0});

    Mat y2 = U * xe;
    Mat x = solve_Ux_y(U, y2);
    assert(is_vector(x));
    assert(x == xe);

    auto b2 = A * xe;
    auto [L2, U2] = LU_decompose(A);
    auto x2 = solve_Ux_y(U2, solve_Ly_b(L2, b2));
    assert(x2 == xe);
}

int main() {
    test_mul();
    test_transpose();
    test_add();
    test_sub();
    test_LU();
    return 0;
}
