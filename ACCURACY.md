# Accuracy experiments

## Comparison to fastText

The goal of these experiments is to test the correctness of
finalfrontier by comparing the results against fastText. The
following hyperparameters are used:

* dims: 300
* mincount: 10
* epochs: 10
* discard: 1e-4
* initial lr: 0.05
* minn/maxn: 3/6
* context: 5
* negative samples: 5
* buckets: 21 (~2,000,000 buckets)
* threads: 16

Computing accuracies:

```
$ compute-accuracy embeddings.txt 30000 < questions-words.txt
```

Note that compute-accuracy needs to be modified to read word2vec
text files:

https://github.com/facebookresearch/fastText/issues/563#issuecomment-413319396

### Wikipedia

| Model                                          | Dataset                 | Semantic Accuracy | Syntactic Accuracy | Total Accuracy |
|------------------------------------------------|-------------------------|-------------------|--------------------|----------------|
| fastText skipgram                              | Wikipedia (en) 20180827 | 85.18             | 76.36              | 79.88          |
| finalfrontier skipgram                         | Wikipedia (en) 20180827 | 85.05             | 75.56              | 79.34          |
| finalfrontier structgram                       | Wikipedia (en) 20180827 | 81.43             | 79.20              | 80.09          |
| final frontier structgram (nonrandom window)   | Wikipedia (en) 20180827 | 82.13             | 78.37              | 79.97          |
| final frontier structgram (only discard focus) | Wikipedia (en) 20180827 | 77.97             | 78.91              | 77.97          |
| final frontier structgram (structured discard) | Wikipedia (en) 20180827 | 77.09             | 79.66              | 78.63          |

### German

Translated Google analogies from:

<http://www.ims.uni-stuttgart.de/forschung/ressourcen/lexika/analogies_ims/analogies.en.html>

`dewiki` is the German Wikipedia of January 2018.

| Model                                          | Dataset      | Semantic Accuracy | Syntactic Accuracy | Total Accuracy |
|------------------------------------------------|--------------|-------------------|--------------------|----------------|
| fastText skipgram                              | taz          | 84.94             | 71.20              | 76.02          |
| fastText skipgram                              | taz + dewiki | 82.00             | 67.01              | 74.24          |
| finalfrontier skipgram                         | taz          | 86.20             | 69.50              | 75.35          |

