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
	$ git clone https://github.com/tensorflow/models.git 
```
Follow the installation instructions [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).
	
```
	# In tensorflow/models/research
	$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim 

	$ python3 script1.py

	$ python3 script2.py

	$ python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record

	$ python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record

	$ bash train.sh

	$ bash export.sh

	$ python3 object_localisation.py
```