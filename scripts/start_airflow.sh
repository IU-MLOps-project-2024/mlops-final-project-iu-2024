#!/bin/bash
# For some weird reason airflow works very strange with --daemon or don't even work with it
airflow triggerer &
airflow scheduler &
airflow webserver &