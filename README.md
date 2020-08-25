# RecSys
- BaselinePredictor
- SVD (SGD / ALS)
- SVD++
- KNN (cosine / pearson / baseline)

## Runtime
### ml-100k (dev:val = 8:2)
| Model                | MSE     | Training Time |
| -------------------- | ------- | ------------- |
| Baseline (20 epochs) | 0.9441  |   4.76 secs   |
| SGD (30 epochs)      | 0.9150  |  16.21 secs   |
| ALS (10 epochs)      | 0.9120  |  17.59 secs   |
| SVDpp (30 epochs)    | 0.9182  | 169.36 secs   |
| KNN                  | 0.9789  | ------------- |

### ml-1m (dev:val = 8:2)
| Model                | MSE     | Training Time |
| -------------------- | ------- | ------------- |
| Baseline (20 epochs) | 0.9120  |  47.31 secs   |
| SGD (30 epochs)      | 0.8650  | 149.21 secs   |
| ALS (10 epochs)      | 0.8586  | 127.37 secs   |
| SVDpp (30 epochs)    | 0.8714  |  48.30 mins   |
| KNN                  | 0.9030  | ------------- |

## Usage
```bash
# download datasets and unzip them to datasets/ folder
# Example: train a matrix factorization model with SGD
cd datasets/
python make_dataset.py --dataset "ml-100k"
cd ..
python --conf confs/ml-100k/svd-sgd.json
```

## Training on your own datasets
- place your csv file as `datasets/dev.csv` and `datasets/val.csv`
- write a config file, picking the model you like 
- run `python main.py --conf <path_to_you_config_file>`

## Misc
The code is highly optimized (JIT / vectorization), but if you're still not satisified with the training speed, see [HERE](https://github.com/Response777/svd) for a cpp implementation of SVD/SVD++, which should be approximately 20x faster. 
- Note that the data format is slightly different.
