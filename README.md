# RecSys
- SVD (SGD / ALS)
- SVD++
- KNN

## Usage
```bash
# python --conf <conf-file>
python --conf confs/svd-sgd.json
```

## Training on your own datasets
- place your csv file as `datasets/raw.csv`
- run `python build_dataset.py` to create dev/val set

### data format for csv file
See `datasets/example.csv`

## Trivia
There is also a Cpp implementation for SVD and SVD++ available (which is approximately 20x faster): https://github.com/Response777/svd

Note that the data format is slightly different.
