# Directory Structure

Setup the following directory structure
```
./
--data
	--training.csv
	--test_labels.csv
--images
```

## Run in the following order
```
	$ python3 script1.py

	$ python3 script2.py

	$ python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record

	$ python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record

	$ bash train.sh

	$ bash export.sh

	$ python3 object_localisation.py
```