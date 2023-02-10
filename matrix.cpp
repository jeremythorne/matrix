#include<cassert>
#include<cstdio>
#include<vector>

// quick and dirty matrix library
// uses asserts rather than error handling

struct Shape {
    size_t M;
    size_t N;
};

template<typename F>
struct Matrix {
    Matrix(Shape s) : shape(s), data(s.M*s.N) {}
    Matrix(Shape s, std::initializer_list<F> list) : shape(s), data(list) {
        assert(shape.M * shape.N == data.size());
    }

    Shape shape;
    const F& at(size_t i, size_t j) const {
        assert(i < shape.M);
        assert(j < shape.N);
        return data[j + i * shape.N];
    }

    F& mut_at(size_t i, size_t j) {
        assert(i < shape.M);
        assert(j < shape.N);
        return data[j + i * shape.N];
    }
private:
    std::vector<F> data;
};

template<typename F>
void print(Matrix<F> const &A) {
    for(size_t i = 0; i < A.shape.M; i++) {
        for(size_t j = 0; j < A.shape.N; j++) {
            printf(" %0.2f", A.at(i, j));
        }
        printf("\n");
    }
}

template<typename F>
F dot(const Matrix<F> &A, size_t row, size_t start_j,
      const Matrix<F> &B, size_t col, size_t start_i, size_t num) {
    assert(row < A.shape.M);
    assert(start_j + num <= A.shape.N);
    assert(col < B.shape.N);
    assert(start_i + num <= B.shape.M);
    F sum{};
    for (size_t i = 0; i < num; i++) {
        sum += A.at(row, start_j + i) * B.at(start_i + i, col);    
    }
    return sum;
}

template<typename F>
Matrix<F> mul(Matrix<F>& A, Matrix<F> &B) {
    assert(A.shape.N == B.shape.M);
    Matrix<F> C({A.shape.M, B.shape.N});
    for(size_t i = 0; i < C.shape.M; i++) {
        for(size_t j = 0; j < C.shape.N; j++) {
            C.mut_at(i, j) = dot(A, i, 0, B, j, 0, A.shape.N);
        }
    }
    return C;
}

using Mat = Matrix<double>;

void test_mul() {
    Mat A({2, 2}, 
            {1.0, 0.0,
             0.0, 1.0});
    Mat B({2, 2}, 
            {2.0, 0.0,
             0.0, 1.0});
    auto C = mul(A, B);
    printf("A:\n");
    print(A);
    printf("B:\n");
    print(B);
    printf("A * B =\n");
    print(C);

    Mat D({2, 1}, {2.0, 
                   3.0});
    printf("D = \n");
    print(D);
    printf("B * D = \n");
    print(mul(B, D));
}

int main() {
    test_mul();
    return 0;
}
