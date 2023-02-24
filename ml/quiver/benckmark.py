import torch
import timeit
import torch.utils.benchmark as benchmark

def batched_dot_mul_sum(a, b):
    '''Computes batched dot by multiplying and summing'''
    return a.mul(b).sum(-1)


def batched_dot_bmm(a, b):
    '''Computes batched dot by reducing to bmm'''
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)


# Input for benchmarking
x = torch.randn(10000, 64)

# Ensure that both functions compute the same output
assert batched_dot_mul_sum(x, x).allclose(batched_dot_bmm(x, x))

'''Benchmark with timeit.Timer'''
# t0 = timeit.Timer(
#     stmt='batched_dot_mul_sum(x, x)',
#     setup='from __main__ import batched_dot_mul_sum',
#     globals={'x': x})
#
# t1 = timeit.Timer(
#     stmt='batched_dot_bmm(x, x)',
#     setup='from __main__ import batched_dot_bmm',
#     globals={'x': x})
#
# print(f'mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us')
# print(f'bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us')


'''Benchmark with torch timer (multi-thread)'''
num_threads = torch.get_num_threads()
print(f'Benchmarking on {num_threads} threads')

t0 = benchmark.Timer(
    stmt='batched_dot_mul_sum(x, x)',
    setup='from __main__ import batched_dot_mul_sum',
    globals={'x': x},
    num_threads=num_threads,
    label='Multithreaded batch dot',
    sub_label='Implemented using mul and sum')

t1 = benchmark.Timer(
    stmt='batched_dot_bmm(x, x)',
    setup='from __main__ import batched_dot_bmm',
    globals={'x': x},
    num_threads=num_threads,
    label='Multithreaded batch dot',
    sub_label='Implemented using bmm')

print(t0.timeit(50))
print(t1.timeit(50))

'''Benchmark with torch timer GPU (multi-thread)'''
x = torch.randn(10000, 1024, device='cuda')

t0 = timeit.Timer(
    stmt='batched_dot_mul_sum(x, x)',
    setup='from __main__ import batched_dot_mul_sum',
    globals={'x': x})

t1 = timeit.Timer(
    stmt='batched_dot_bmm(x, x)',
    setup='from __main__ import batched_dot_bmm',
    globals={'x': x})


# Ran each twice to show difference before/after warmup
print(f'mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us')


m0 = t0.blocked_autorange()
m1 = t1.blocked_autorange()

print(m0)
print(m1)