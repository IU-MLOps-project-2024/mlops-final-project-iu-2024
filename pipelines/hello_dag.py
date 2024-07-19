# pipelines/hello_dag.py

from datetime import datetime

from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator

# A DAG represents a workflow, a collection of tasks
# This DAG is scheduled to print 'hello world' every minute starting from 01.01.2022.
with DAG(dag_id="hello_world", 
		 start_date=datetime(2024, 7, 18, 18, 55, 0),
		 schedule="*/5 * * * *") as dag:
	
    # Tasks are represented as operators
    # Use Bash operator to create a Bash taskpi
    hello = BashOperator(task_id="hello", bash_command="echo hello")
	
    # Python task
    @task()
    def world():
        print("world")
		
    # Set dependencies between tasks
    # First is hello task then world task
    hello >> world()