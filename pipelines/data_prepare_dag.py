from pendulum import datetime
from datetime import timedelta

from airflow import DAG
from airflow.decorators import dag
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule


with DAG(dag_id="data_prepare_dag",
         start_date=datetime(2024, 7, 24, tz="UTC"),
         schedule="*/10 * * * *",
         catchup=False) as dag:

    data_extract = TriggerDagRunOperator(
        task_id='trigger_data_extract_dag',
        trigger_dag_id='data_extract_dag',
        start_date=datetime(2024, 7, 24)
    )

    run_zenml_pipeline = BashOperator(
        task_id='run_zenml_pipeline',
        bash_command='cd ~/Desktop/mlops-final-project-iu-2024/pipelines; zenml pipeline run data_prepare_pipeline ',
        trigger_rule=TriggerRule.ALL_SUCCESS,
        dag=dag,
    )

    data_extract >> run_zenml_pipeline

if __name__ == "__main__":
    dag.test()
