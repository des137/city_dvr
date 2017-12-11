from util import benchmark
import time

def test_benchmark():
    with benchmark('foo'):
        time.sleep(0.1)
