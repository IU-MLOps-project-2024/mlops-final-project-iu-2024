# Sample the data and version it
python ./src/data.py
dvc add data/samples/sample.csv
dvc push

# Validate the data
python ./src/gx_checkpoint.py