# A Supervised Multi-Head Self-Attention Network for Nested Named Entity Recognition

## Dependecies and installation

The main dependencies are:
- [AllenNLP 0.9.0](https://github.com/allenai/allennlp)
- [PyTorch 0.4.1](https://pytorch.org/)

## Datasets

### 1. Download the data

ACE-2004:[https://catalog.ldc.upenn.edu/LDC2005T09](https://catalog.ldc.upenn.edu/LDC2005T09)
ACE-2005:[ https://catalog.ldc.upenn.edu/LDC2006T06]( https://catalog.ldc.upenn.edu/LDC2006T06)
GENIA:[ http://www.geniaproject.org/]( http://www.geniaproject.org/)
JNLPBA-train:[http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Train/Genia4ERtraining.tar.gz](http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Train/Genia4ERtraining.tar.gz)
JNLPBA-test:[http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Train/Genia4ERtraining.tar.gz](http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Evaluation/Genia4ERtest.tar.gz)
CoNLL03-English:[https://www.clips.uantwerpen.be/conll2003/ner/](https://www.clips.uantwerpen.be/conll2003/ner/)

We follow the script from [https://github.com/thecharm/boundary-aware-nested-ner](https://github.com/thecharm/boundary-aware-nested-ner) to preprocess the corpora.

After proprecessed, we can get the processed files for training, like, `genia.train.raw.pkl/genia.test.raw.pkl/genia.dev.raw.pkl`
readding data:
```
import joblib
sentences,records=joblib.load("genia.train.raw.pkl")
sentences:[[sentence1],[sentence2],...[sentenceN]],N is the number of the sentences in each file.
records is a list of dicts：[{(3,5):"DNA",(6,7):"RNA",(6,10):"DNA"...},{},...{}]
```
### download pre-trained bert

Download [BERT-BASE-UNCASED](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz) and its [vocab.txt](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt) , and put it under data/bert/

## Run
We based our implementation on the [AllenNLP library](https://github.com/allenai/allennlp). For an introduction to this library, you should check [these tutorials](https://allennlp.org/tutorials).

An experiment is defined in a _json_ configuration file (see `configs/*.json` for examples). The configuration file mainly describes the datasets to load, the model to create along with all the hyper-parameters of the model. 

Once you have set up your configuration file (and defined custom classes such `DatasetReaders` if needed), you can simply launch a training with the following command and arguments:

```bash
python train_json.py --config_file_path configs/smhsan-genia.json --serialization_dir my_genia_training
```


## References

Please consider citing the following paper if you find this repository useful.
```


@inproceedings{XuHF021,
  author    = {Yongxiu Xu and
               Heyan Huang and
               Chong Feng and
               Yue Hu},
  title     = {A Supervised Multi-Head Self-Attention Network for Nested Named Entity
               Recognition},
  booktitle = {Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI}
               2021, Thirty-Third Conference on Innovative Applications of Artificial
               Intelligence, {IAAI} 2021, The Eleventh Symposium on Educational Advances
               in Artificial Intelligence, {EAAI} 2021, Virtual Event, February 2-9,
               2021},
  pages     = {14185--14193},
  publisher = {{AAAI} Press},
  year      = {2021},
}


```
