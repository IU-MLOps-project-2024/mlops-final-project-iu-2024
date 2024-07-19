from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import sample_data
from src.gx_checkpoint import validate_initial_data

import subprocess
import yaml

def validate_sample(**kwargs):
    validate_initial_data('~/Desktop/mlops-final-project-iu-2024/data/samples/sample.csv')

def load_sample_to_dvc_remote(**kwargs):
    subprocess.run(
        [
            "dvc",
            "add",
            '~/Desktop/mlops-final-project-iu-2024/data/samples/sample.csv'
        ],
        check=True
    )
    subprocess.run(
        [
            "dvc",
            "remote",
            "add",
            "-d",
            "localstore",
            "~/Desktop/mlops-final-project-iu-2024/datastore"
        ],
        check=True
    )
    subprocess.run(["dvc", "push"], check=True)

def version_sample(
    data_path='data/samples/sample.csv',
    version_file='~/Desktop/mlops-final-project-iu-2024/configs/data_version.yaml',
    **kwargs
):
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            version_data = yaml.safe_load(f)
            current_version = version_data.get('version', 0)
    else:
        current_version = 0
    
    new_version = current_version + 1

    # run_command(['dvc', 'add', data_path])
    subprocess.run(['dvc', 'add', data_path], check=True)
    subprocess.run(['git', 'add', data_path + '.dvc'], check=True)
    # subprocess.run(['git', 'commit', '-m', f"Version {new_version} of sample data"], check=True)
    run_command(['git', 'commit', '-m', f"Version {new_version} of sample data"])

    subprocess.run(['dvc', 'push'], check=True)

    version_data = {'version': new_version}
    with open(version_file, 'w') as f:
        yaml.safe_dump(version_data, f)


def run_command(command):
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command)}")
        print(f"Exit status: {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        raise

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'data_extract_dag',
    default_args=default_args,
    description='A simple data extraction pipeline',
    schedule_interval=timedelta(minutes=5),
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

# Task 1: Extract a new sample of the data
extract_task = PythonOperator(
    task_id='extract_sample',
    python_callable=sample_data,
    dag=dag,
)

# Task 2: Validate the sample using Great Expectations
validate_task = PythonOperator(
    task_id='validate_sample',
    python_callable=validate_sample,
    dag=dag,
)

# version_step = BashOperator(
#     task_id= 'version_data',
#     bash_command="../scripts/test_data.sh ../data/samples/sample.csv dev",
#     dag=dag
# )

# Task 3: Version the sample using DVC
version_task = PythonOperator(
    task_id='version_sample',
    python_callable=version_sample,
    dag=dag,
)

# Task 4: Load the sample to the data store
load_task = PythonOperator(
    task_id='load_sample_to_dvc_remote',
    python_callable=load_sample_to_dvc_remote,
    dag=dag,
)

# Set up the task dependencies
extract_task >> validate_task >> version_task >> load_task

if __name__ == '__main__':
    dag.test()
