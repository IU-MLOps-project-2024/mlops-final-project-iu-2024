#!/bin/bash
# For some weird reason airflow works very strange with --daemon or don't even work with it
airflow triggerer --log-file services/airflow/logs/triggerer.log &
airflow scheduler --log-file services/airflow/logs/scheduler.log &
airflow webserver --log-file services/airflow/losg/webserver.log &