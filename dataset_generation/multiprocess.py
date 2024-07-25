import multiprocessing
import subprocess
import sys
import os
import numpy as np

def run_script(args):
    """
    Function to run the script with given arguments.
    """
    script_path = "dataset_generation/create_dataset.py"
    command = [sys.executable, script_path] + args
    subprocess.run(command)
    print("done")

if __name__ == "__main__":
    # List of argument sets for each instance
    #step = 100
    #starts = list(range(0, 2301, step))
    #intervals = [[s, s+step] for s in starts]
    '''
    l = []
    for i in range(2454):
        data_path = '/data/Datasets/nuscenes_custom/data/actions/'
        path = data_path + str(i) + '.pt'
        if not os.path.exists(path):
            l.append(i)

    step = 10 
    intervals = []
    for i in range(0, len(l), step):
        intervals.append([l[i], l[min(i+step, len(l)-1)]])
    '''

    intervals = [[134, 135], [234, 235], [255,256]]

    # Create a pool with the number of CPUs you want to use
    num_processes = min(61, len(intervals))
    pool = multiprocessing.Pool(processes=num_processes)

    # Start each process
    for interval in intervals:
        args = ["--overwrite", "True", "--version", "v1.0-trainval", "--start_idx", str(interval[0]), "--end_idx", str(interval[1])]
        print(args)
        pool.apply_async(run_script, (args,))

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    print("All processes completed")