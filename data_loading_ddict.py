import dragon
from multiprocessing import set_start_method, Pool, cpu_count, current_process
from pathlib import Path

from dragon.data.ddict import DDict
from dragon.native.machine import System, Node


def initialize_worker(the_ddict):
    # Since we want each worker to maintain a persistent handle to the DDict,
    # attach it to the current/local process instance. Done this way, workers attach only
    # once and can reuse it between processing work items

    me = current_process()
    me.stash = {}
    me.stash["ddict"] = the_ddict


def process_data(idx_size):
    the_ddict = current_process().stash["ddict"]
    try:
        idx, size = idx_size
        k = f"some_key_{idx}"
        v = bytearray(size)
        the_ddict[k] = v
        return True
    except Exception as e:
        return e


def setup_ddict():
    # let's place the DDict across all nodes Dragon is running on
    my_system = System()
    num_nodes = my_system.nnodes

    total_mem = 0
    for huid in my_system.nodes:
        anode = Node(huid)
        total_mem += anode.physical_mem
    dict_mem = 0.1 * total_mem  # use 10% of the mem
    print(f"DDict will be {dict_mem} bytes", flush=True)

    return DDict(
        2,  # two managers per node
        num_nodes,
        int(dict_mem),
    )


class DataReader:

    def __init__(self, items_mb, nitems):
        self.size = int(items_mb * 1024**2)
        self.nitems = nitems
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        if self.idx >= self.nitems:
            raise StopIteration
        return (self.idx, self.size)


if __name__ == "__main__":
    set_start_method("dragon")

    items_mb = 50
    nitems = 1000
    dr = DataReader(items_mb, nitems)

    the_ddict = setup_ddict()

    num_cores = cpu_count() // 8

    with Pool(num_cores, initializer=initialize_worker, initargs=(the_ddict,)) as p:
        processed_data = p.imap_unordered(process_data, dr, 64)
        for i, item in enumerate(processed_data):
            if item != True:
                print(f"Worker caught an exception: {item}", flush=True)
            if (i % 100) == 0:
                print(f"loaded {i} of {nitems}", flush=True)

    the_ddict.destroy()
