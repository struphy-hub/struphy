from numpy import zeros, ones
from numpy import int32
from pyccel.stdlib.internal.openmp import omp_set_num_threads, omp_get_num_threads, omp_get_thread_num


def f(n : int):
    a = ones((n))
    x = ones((n))
    
    omp_set_num_threads(int32(n))
    #$ omp parallel shared(x)
    thread_num = omp_get_thread_num()
    num_threads = omp_get_num_threads()
    # print("")
    # print("hello from thread number:", thread_num, "out of", num_threads)
    x[thread_num] = thread_num
    x[thread_num] += a[thread_num]
    #$ omp end parallel
    return x

def g(n : int):
    a = ones((n))
    x = ones((n))
    omp_set_num_threads(int32(n))
    #$ omp parallel shared(x) private(thread_num)
    thread_num = omp_get_thread_num()
    num_threads = omp_get_num_threads()
    print("hello from thread number:", thread_num, "out of", num_threads)
    x[thread_num] = thread_num
    x[thread_num] += a[thread_num]
    #$ omp end parallel
    return x



if __name__ == '__main__':
    n = 10
    x1 = f(n)
    x2 = g(n)
    
    # Should print [1.000000000000 2.000000000000 3.000000000000 4.000000000000, ...]
    print(x1)
    print(x2)
