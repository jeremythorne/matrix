CXXFLAGS=-std=c++17 -g

test: test.o matrix.h
	g++ ${CXXFLAGS} -o test test.o

clean:
	rm *.o test
