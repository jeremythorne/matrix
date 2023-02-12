CXXFLAGS=-std=c++17
test: test.o matrix.h
	g++ ${CXXFLAGS} -o test test.o
