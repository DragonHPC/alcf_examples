import dragon
from multiprocessing import set_start_method, Pool, cpu_count, current_process
from pathlib import Path

from dragon.data.ddict import DDict
from dragon.native.machine import System, Node

import sys


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

        smiles = []
        inf_results = []
        tot_size = 0
        while tot_size < size:
            smiles.append(64 * "a")
            inf_results.append(0.0)
            tot_size = sys.getsizeof(smiles) + sys.getsizeof(inf_results)

        v = {"f_name": k, "smiles": smiles, "inf": inf_results, "model_iter": -1}
        # v = bytearray(size)
        the_ddict[k] = v
        return True
    except Exception as e:
        return e


def setup_ddict(needed_mb):
    # let's place the DDict across all nodes Dragon is running on
    my_system = System()
    num_nodes = my_system.nnodes
    node_mem_frac = 0.5
    head_room_fac = 2

    total_mem = 0
    for huid in my_system.nodes:
        anode = Node(huid)
        total_mem += anode.physical_mem
    dict_mem = min(head_room_fac * needed_mb * 1024**2, node_mem_frac * total_mem)
    print(
        f"DDict will be {int(dict_mem / 1024**3)} GB (needed {int(head_room_fac * needed_mb / 1024)} GB)",
        flush=True,
    )

    return DDict(
        1,
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

    items_mb = 14.6
    nitems = 500354  # 10000 * System().nnodes
    dr = DataReader(items_mb, nitems)

    the_ddict = setup_ddict(items_mb * nitems)

    num_cores = cpu_count() // 8

    with Pool(num_cores, initializer=initialize_worker, initargs=(the_ddict,)) as p:
        processed_data = p.imap_unordered(process_data, dr, chunksize=64)
        for i, item in enumerate(processed_data):
            if item != True:
                print(f"Worker caught an exception: {item}", flush=True)
            if (i % 100) == 0:
                print(f"loaded {i} of {nitems}", flush=True)

    the_ddict.destroy()
