# Build skeleton only
python main.py skeleton

# Run single task
python main.py run text
python main.py run img_desc
python main.py run cta
python main.py run audience

# Run task on existing CSV
python main.py run audience dataset.csv

# Full pipeline (all tasks in sequence)
python main.py generate

# Full pipeline, ignore saved progress
python main.py generate --fresh

# Check progress
python main.py status

# Clean up progress files
python main.py clean

# Validate a dataset
python main.py validate generated_datasets/dataset.csv

# List available tasks
python main.py tasks