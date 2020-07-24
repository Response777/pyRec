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
The code is highly optimized (JIT / vectorization), but if you're still not satisified with the training speed, see [HERE](https://github.com/Response777/svd) for a cpp implementation of SVD/SVD++, which should be approximately 20x faster. 
- Note that the data format is slightly different.
