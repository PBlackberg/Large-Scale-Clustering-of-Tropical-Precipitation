'''
# ----------------------
#  interactive_session
# ----------------------

'''

import subprocess

def start_interactive_session():
    try:
        command = [
            "salloc",
            "--account=bb1153",
            "--partition=interactive",
            "--time=4:00:00",
            "--nodes=1",
            "--exclusive",
            # "--ntasks=1",
            # "--cpus-per-task=64",

        ]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting interactive session: {e}")
    except KeyboardInterrupt:
        print("Interactive session cancelled.")

if __name__ == "__main__":
    start_interactive_session()

    # scontrol show job $SLURM_JOB_ID
