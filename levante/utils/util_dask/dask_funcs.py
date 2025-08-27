'''
# ----------------
#   Dask funcs
# ----------------

'''

# == imports ==
# -- Packages --
from distributed import Client, LocalCluster, get_client
import multiprocessing
import time
from functools import wraps
import dask


# == key dask funcs ==
def create_dask_cluster(nworkers, cpus_per_worker = 'use_all', memory_per_worker = 'default', print_worker = True):
    dask.config.set({"distributed.scheduler.worker-ttl": "10s"})
    ncpus = multiprocessing.cpu_count()
    cpus_per_worker = (ncpus // nworkers) // 2 * 2              if cpus_per_worker == 'use_all'     else cpus_per_worker # even number of cpus / worker (so a core don't need to be split across workers) - maybe doesn't matter
    memory_per_worker = f'{int(0.9 * cpus_per_worker)}GB'       if memory_per_worker == 'default'   else memory_per_worker
    cluster = LocalCluster(n_workers            = nworkers, 
                           threads_per_worker   =cpus_per_worker, 
                           memory_limit         = memory_per_worker)
    print(f'-- Created dask client with {nworkers} workers --')
    client = Client(cluster)
    if print_worker:
        print(f'  -- Worker Specs --')  
        workers = client.scheduler_info()['workers']
        for i, (worker, info) in enumerate(workers.items()):
            print(f'  worker {i}')
            print(f"  Name: {worker}")
            print(f"  Memory Limit: {info['memory_limit'] / 1e9:.2f} GB")
            print(f"  Threads: {info['nthreads']}")
            print(f"  CPU Utilization: {info['metrics']['cpu']:.2f}%")
            print(f"  Memory Usage: {info['metrics']['memory'] / 1e9:.2f} GB")
            print("")
            break

def apply_dask_delayed(decorator = dask.delayed, apply_decorator = False):
    def decorator_wrapper(func):
        if apply_decorator:
            return decorator(func)
        return func
    return decorator_wrapper


def open_dask_website():
    import webbrowser
    client = get_client()
    print(client.dashboard_link)
    exit()
    webbrowser.open(f'{client.dashboard_link}') 

# == debug worker ==
def get_worker(worker_nb = 0, print_worker = True):
    client = get_client()
    workers = list(client.scheduler_info()['workers'].keys())[worker_nb]
    if print_worker:
        workers = client.scheduler_info()['workers']
        for i, (worker, info) in enumerate(workers.items()):
            if i == worker_nb: 
                print('Requested one worker:')
                print(f'  -- Worker Specs --')  
                print(f'  Worker {i}')
                print(f"  Name: {worker}")
                print(f"  Memory Limit: {info['memory_limit'] / 1e9:.2f} GB")
                print(f"  Threads: {info['nthreads']}")
                print(f"  CPU Utilization: {info['metrics']['cpu']:.2f}%")
                print(f"  Memory Usage: {info['metrics']['memory'] / 1e9:.2f} GB")
                print('')
                break
    return worker

def execute_with_one_worker(worker, delayed_task):
    client = get_client()
    result = client.compute(delayed_task, workers=[worker]).result()
    return result

def scatter_data(data_object):
    ''' distribute data to workers  '''
    client = get_client()
    return client.scatter(data_object, direct=True, broadcast=True)     

def monitor_worker(worker, client, interval=2.5, log_file="worker_monitor.log"):
    """Monitor CPU and memory usage of a specific worker and log to a file."""
    with open(log_file, "w") as f:
        f.write(f"Monitoring worker {worker}...\n")
        while True:
            workers_info = client.scheduler_info()['workers']
            if worker not in workers_info:
                f.write("Worker no longer available. Exiting monitoring.\n")
                break
            metrics = workers_info[worker]['metrics']
            log_entry = (
                f"CPU Utilization: {metrics['cpu']:.2f}%, "
                f"Memory Usage: {metrics['memory'] / 1e9:.2f} GB\n"
            )
            f.write(log_entry)
            f.flush()  # Ensure the logs are written immediately
            time.sleep(interval)


# == handle client ==
def close_dask_cluster_after(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print('closing dask cluster')
        get_client().close()
        return result
    return wrapper

def rescale_dask_client(nworkers=1):
    client = get_client()
    client.cluster.scale(nworkers)
    print(f"Rescaled Dask client to {nworkers} worker(s).")


    


if __name__ == '__main__':
    print('testing dask funcs')
    nworkers = 6
    create_dask_cluster(nworkers, cpus_per_worker = 'use_all', memory_per_worker = 'default', print_worker = True)
    get_client().close()
    print('finished')


