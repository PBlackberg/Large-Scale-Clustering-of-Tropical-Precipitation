'''
# ---------------------
#  submission_funcs
# ---------------------

'''

# == imports ==
# -- Packages --
import os
import subprocess
import time


# == key submission funcs ==
def create_resource_script(python_script, folder, filename, walltime, env_variables, SU_project,
                           partition = 'compute', 
                           mem = None,
                           ncpus = None,
                           exclusive = False,
                           dependencies = None
                           ):
    command_line = f'python $PYTHON_SCRIPT $SLURM_SCRIPT'
    if env_variables:
        command_line += " " + " ".join([f"${key}" for key, value in env_variables.items()])
    mem_line        = f'#SBATCH --mem={mem}'                                                if mem                      else ''
    ncpus_line      = f'#SBATCH --cpus-per-task={ncpus}'                                    if ncpus                    else ''
    exclusive_line  = f'#SBATCH --exclusive'                                                if exclusive                else ''
    dependency_line = f'#SBATCH --dependency=afterok:{":".join(map(str, dependencies))}'    if dependencies             else ''

    slurm_script = f'{folder}/{filename}.slurm'
    oe_path = f'{folder}/{filename}.out'
    os.makedirs(folder, exist_ok=True)

    script_content = f"""#!/bin/bash
#SBATCH --account={SU_project}
#SBATCH --job-name={filename}
#SBATCH --output={oe_path}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --time={walltime}
{mem_line}
{ncpus_line}
{exclusive_line}
{dependency_line}""".strip() + f"""
echo "== Starting {filename} =="
echo "-- Loading modules --"
module load python3/2023.01-gcc-11.2.0
module use /work/k20200/k202134/hsm-tools/outtake/module
module load hsm-tools/unstable
echo ""
echo "-- Job Specs --"
scontrol show job $SLURM_JOB_ID
PYTHON_SCRIPT={python_script}
{command_line}"""
    with open(slurm_script, 'w') as file:
        file.write(script_content)
    # print('\tcreated resource script')
    return slurm_script, oe_path

def submit_job_slurm(python_script, folder, filename, walltime, env_variables, SU_project,
                           partition = 'compute', 
                           mem = None,
                           ncpus = None,
                           exclusive = False,
                           dependencies = None):
    slurm_script, oe_path = create_resource_script(python_script, folder, filename, walltime, env_variables, SU_project, partition, mem, ncpus, exclusive, dependencies)
    env_variables[f"SLURM_SCRIPT"] = slurm_script
    env_vars_str = ",".join([f"{key}={value}" for key, value in env_variables.items()])
    command = [
        "sbatch", 
        f"--export={env_vars_str}",
        slurm_script
        ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
    job_id = result.stdout.strip().split()[-1]  # Extract job ID from sbatch output
    # print(f"\tSubmitted job\n\tOutput/Error file:\n\t{oe_path}\n\tJob ID: {job_id}\n")
    # print(f"Submitted job..") # Output/Error file at: {oe_path}")
    return job_id


# == funcs to handle job queue and variables given to job ==
def check_squeue(user, delay=5, watch = False):
    time.sleep(delay)   # sleep to let queue update
    try:
        command = ['squeue', '-u', user]
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
        total_jobs = len(result.stdout.splitlines()) - 1  # Subtracting 1 for the header
        print(f'Total number of jobs in queue: {total_jobs}')
        if total_jobs > 900:
            print(f'jobs in queue are close to the limit {total_jobs} / 1000')
            print('wait for completion of other jobs')
            print('exiting')
            exit()
    except subprocess.CalledProcessError as e:
        print("Error:")
        print(e.stderr)
        
def switch_to_env_variables(env_variables, switch):
    start_idx = len(env_variables)  # when options are updated, previous options shouldn't take the smae place
    env_variables.update({f"GIVEN_OPT{start_idx + i}": key for i, (key, value) in enumerate(switch.items()) if value})

def check_no_conflicting_env_variables(env_variables):
    """Check for conflicts with existing environment variables."""
    conflicts = {key: value for key, value in env_variables.items() if key in os.environ}
    if conflicts:
        print("Warning: The following environment variables already exist:")
        [print(key) for key, value in conflicts.items()]
        print('rename them to something else')
        print('exiting')
        exit()


if __name__ == '__main__':
    user = 'b382628'
    check_squeue(user, delay=5)









