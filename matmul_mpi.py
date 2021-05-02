"""
MPI-accelerated matrix multiplication python code

Rangsiman Ketkaew
"""


from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
computeSize = comm.Get_size() - 1
rank = comm.Get_rank()
processor_name = MPI.Get_processor_name()

print(f"Rank {rank} : of Nodes {computeSize} --> processor {processor_name}")

elemRow = 3000
elemCol = 3000

def multiply_matrix(a, b):
    return np.matmul(a, b)


def split_matrix(a, b):
    """Split matrix according to the number of compute nodes and send them to
    compute nodes separately.
    """
    rows = []
    n = len(a) // computeSize
    r = len(a) % computeSize
    d, e = 0, n + min(1, r)
    for i in range(computeSize):
        rows.append(a[d:e])
        r = max(0, r - 1)
        d, e = e, e + n + min(1, r)

    # send each matrix to compute node
    pid = 1
    for row in rows:
        comm.send(row, dest=pid, tag=1)
        comm.send(b, dest=pid, tag=2)
        pid = pid + 1

def compute_node_operation():
    # receive data from master node
    a = comm.recv(source=0, tag=1)
    b = comm.recv(source=0, tag=2)
    # multiply the received matrix and send the result back to master
    c = multiply_matrix(a, b)
    # send result back to master node
    comm.send(c, dest=0, tag=rank)
    return c

def collect_matrix():
    """Compile values that returns from compute nodes.
    """
    mat = np.array([]).reshape(0, elemCol)
    pid = 1
    for n in range(computeSize):
        row = comm.recv(source=pid, tag=pid)
        mat = np.vstack((mat, row))
        pid = pid + 1
    return mat


if __name__ == '__main__':
    if rank == 0:
        a = np.random.randint(0, 100, size=(elemRow, elemCol))
        b = np.random.randint(0, 100, size=(elemRow, elemCol))
        print(f"\nInput matrix: {a.shape} x {b.shape}")

        # start time
        t1 = time.time()

        split_matrix(a, b)
        c = collect_matrix()
        print(f"Output matrix: {c.shape}")

        # end time
        t2 = time.time()

        print(f"Elapsed time: {t2 - t1} second")
    else:
        c = compute_node_operation()


