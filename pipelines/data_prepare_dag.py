from datetime import datetime, timedelta
from airflow import DAG
from airflow.sensors.external_task_sensor import ExternalTaskSensor
from airflow.operators.bash_operator import BashOperator
from airflow.utils.trigger_rule import TriggerRule

# Define default_args
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'data_prepare_dag',
    default_args=default_args,
    description='DAG to run ZenML pipeline after data extraction is successful',
    schedule_interval=timedelta(minutes=5),
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    # Task 1: ExternalTaskSensor to wait for completion of data_extract_dag
    wait_for_data_extraction = ExternalTaskSensor(
        task_id='wait_for_data_extraction',
        external_dag_id='data_extract_dag',
        external_task_id=None,  # Wait for all tasks in the external DAG to complete
        allowed_states=['success'],
        failed_states=['failed', 'skipped'],
        execution_delta=timedelta(minutes=5),
        mode='poke',
        timeout=600,
        poke_interval=30,
        retries=2,
    )

    # Task 2: BashOperator to run the ZenML pipeline
    run_zenml_pipeline = BashOperator(
        task_id='run_zenml_pipeline',
        bash_command='zenml pipeline run data_prepare_pipeline',
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Set task dependencies
    wait_for_data_extraction >> run_zenml_pipeline


