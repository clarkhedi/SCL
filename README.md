# Expert-Guided Cross-Modal Semantic Consensus Learning for Text-to-Image Person Re-identification

![main](doc/scrnet.png)



### Getting started

Install dependencies

```
pip install -r requirements.txt
```

Any components in NLTK can be found [here](https://github.com/nltk/nltk_data/tree/gh-pages).



### Prepare pretrained model

Download pretrained baseline from [model_base](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth ) from [BLIP](https://github.com/salesforce/BLIP) and put **model_base.pth** at **./checkpoint**

### Prepare datasets

Download datasets: [CUHK-PEDES](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description)   [ICFG-PEDES](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description)  [RSTPReid](https://github.com/NjtechCVLab/RSTPReid-Dataset) and put at **./datasets**

The files shall be organized as:

```
|---configs
|---data
|---models
|---datasets
|	---CUHK-PEDES
|	---ICFG-PEDES
|	---RSTPReid
|---checkpoint
|	---model_base.pth
|
```



### Train

```
bash train.sh
```



If you have problem in downloading pretrained BERT models here:

```
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
```

you may manually download it from [hugging face](https://huggingface.co/docs/transformers/model_doc/bert#bert) and use this instead:

```
tokenizer = BertTokenizer.from_pretrained(bert_localpath)
```



### Evaluation

```
bash eval.sh
```



### Acknowledgments

Sincerely appreciate the contributions of [BLIP](https://github.com/salesforce/BLIP), [NLTK](https://github.com/nltk/nltk_data/tree/gh-pages), and [IRRA](https://github.com/anosorae/IRRA), for we used codes and pretrained models from these previous works.
