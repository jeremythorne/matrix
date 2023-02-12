#ifndef MATRIX_H
#define MATRIX_H

#include<cassert>
#include<vector>
#include <cstdio>

// quick and dirty matrix library
// uses asserts rather than error handling

// dimensions of a Matrix
struct Shape {
    size_t M;
    size_t N;
};

bool operator==(Shape const& a, Shape const &b) {
    return a.M == b.M && a.N == b.N;
}

// a heap allocated matrix
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

    Matrix<F> t() const {
        return transpose(*this);
    }
private:
    std::vector<F> data;
};

using Mat = Matrix<double>;

// print A to stdout
template<typename F>
void print(Matrix<F> const &A) {
    for(size_t i = 0; i < A.shape.M; i++) {
        for(size_t j = 0; j < A.shape.N; j++) {
            printf(" %0.2f", A.at(i, j));
        }
        printf("\n");
    }
}

// calculate sum of products between sub row of A and sub column of B
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

// calculate A * B
template<typename F>
Matrix<F> mul(Matrix<F> const &A, Matrix<F> const &B) {
    assert(A.shape.N == B.shape.M);
    Matrix<F> C({A.shape.M, B.shape.N});
    for(size_t i = 0; i < C.shape.M; i++) {
        for(size_t j = 0; j < C.shape.N; j++) {
            C.mut_at(i, j) = dot(A, i, 0, B, j, 0, A.shape.N);
        }
    }
    return C;
}

template<typename F>
Matrix<F> operator*(Matrix<F> const& A, Matrix<F> const &B) {
    return mul(A, B);
}

// calculate A == B
template<typename F>
bool eq(Matrix<F>const &A, Matrix<F> const &B) {
    assert(A.shape == B.shape);

    for(size_t i = 0; i < A.shape.M; i++) {
        for(size_t j = 0; j < A.shape.N; j++) {
            if (A.at(i, j) != B.at(i, j)) {
                return false;
            }
        }
    }
    return true;
}

template<typename F>
bool operator==(Matrix<F> const& A, Matrix<F> const &B) {
    return eq(A, B);
}

// transpose A
template<typename F>
Matrix<F> transpose(Matrix<F> const &A) {
    Matrix<F> B({A.shape.N, A.shape.M});
    for(size_t i = 0; i < A.shape.M; i++) {
        for(size_t j = 0; j < A.shape.N; j++) {
            B.mut_at(j, i) = A.at(i, j);
        }
    }
    return B;
}

#endif
