from datetime import datetime, timedelta
from airflow import DAG
from airflow.sensors.external_task_sensor import ExternalTaskSensor
from airflow.operators.bash_operator import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


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
    start_date=datetime(2024, 7, 24),
    catchup=False,
) as dag:

    wait_for_data_extraction = ExternalTaskSensor(
        task_id='wait_for_data_extraction',
        external_dag_id='data_extract_dag',
        external_task_id="load_sample_to_dvc_remote",
        timeout=300
    )

    run_zenml_pipeline = BashOperator(
        task_id='run_zenml_pipeline',
        bash_command='zenml pipeline run data_prepare_pipeline',
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Set task dependencies
    wait_for_data_extraction >> run_zenml_pipeline

if __name__ == "__main__":
    dag.test()


