from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import sample_data
from src.data import validate_sample
from src.data import version_sample
from src.gx_checkpoint import validate_initial_data

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
    start_date=datetime(2024, 7, 24),
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
    trigger_rule='all_success',
    dag=dag,
)

# Task 3: Version the sample using DVC
version_task = PythonOperator(
    task_id='version_sample',
    python_callable=version_sample,
    trigger_rule='all_success',
    dag=dag,
)

# Task 4: Load the sample to the data store
load_task = BashOperator(
    task_id='load_sample_to_dvc_remote',
    bash_command='~/Desktop/mlops-final-project-iu-2024/scripts/load_sample_to_dvc_remote.sh ',
    trigger_rule='all_success',
    dag=dag,
)


# Set up the task dependencies
extract_task >> validate_task >> version_task >> load_task

if __name__ == '__main__':
    dag.test()
