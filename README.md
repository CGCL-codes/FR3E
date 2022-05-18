# Financial text-oriented Entity Extraction and Relation Extraction model (FR3E)
This project is based on [CasRel](https://github.com/Onion12138/CasRelPyTorch) and [PURE](https://github.com/princeton-nlp/PURE). This model implements the financial text-oriented entity extraction and relation extraction model to solve the problems of information extraction from financial texts:
- Entity extraction on financial texts has the problem of nested entities. This project adopts a span-level entity extraction structure to solve it.
- Relation extraction on financial texts has the problem of complex relation types. This project adopts a cascade tagging network structure, which first predicts relation subjects and then predicts the corresponding relation objects under each relation type.
- Relation extraction model does not fully make use of the result of entity extraction model. In order to make full use of the entity type feature, the entity type label predicted by the model is added to the text and encoded, and fused with the entity vector as the input of the relation extraction model, so that the two models can be effectively combined.

## Entity Extraction Model
Entity extraction model implements the span-level entity extraction structure so that the model can extract nested entities effectively.

### Requirements
- tqdm
- allennlp==0.9.0
- torch==1.4.0
- transformers==3.0.2
- overrides==3.1.0
- requests==2.25.1

### Usage
The finance dataset is built by human annotations. Since it is not published yet, this project only provides several instances. You can run the model on other open source datasets. 

Input data format for the entity model is the same as PURE. Since this project does not consider the cross-sentence feature, every document instance only contains one sentence.
```
{"doc_key": "249", 
"sentences": [
    ["报", "告", "期", "内", "，", "中", "船", "科", "技", "所", "属", "行", "业", "为", "建", "筑", "业", "，", ...]
], 
"ner": [
    [[5, 8, "公司企业"], [14, 16, "行业"], [21, 24, "公司企业"], [43, 44, "业务"]]
],
"relations": [
    [[5, 8, 14, 16, "属于"], [21, 24, 5, 8, "投资"], [5, 8, 43, 44, "开展"]]
]
}
```
Use `run_entity.py` with `--do_train` to train an entity model and with `--do_eval` to evaluate an entity model.
```
python run_entity.py \
    --do_train --do_eval [--eval_test] \
    --learning_rate=1e-5 --task_learning_rate=5e-4 \
    --train_batch_size=batch_size \
    --task task_name \
    --data_dir {directory of preprocessed dataset} \
    --model model_name \
    --output_dir {directory of output files}
```
Arguments:
* `--learning_rate`: the learning rate for BERT encoder parameters.
* `--task_learning_rate`: the learning rate for task-specific parameters, i.e., the classifier head after the encoder.
* `--batch_size`: the batch size when training the model.
* `--model`: the base transformer model. 
* `--eval_test`: whether evaluate on the test set or not.

The predictions of the entity model will be saved in `ent_pred_dev.json` and `ent_pred_test.json` for development set and test set. The prediction file of the entity model will be used to add entity type markers in the input file of the relation model.

## Relation Extraction Model
Relation extraction model utilizes the predicted result of entity extraction model by adding the predicted entity type markers in the text. Besides, the model adopts the cascade tagging network structure to extract complex relation types.
### Requirements
- torch==1.8.0+cu111
- transformers==4.3.3
- fastNLP==0.6.0
- tqdm==4.59.0
- numpy==1.20.1

### Usage
The finance dataset is built by human annotations. Since it is not published yet, this project only provides several instances. You can run the model on other open source datasets. 

Input data format is shown below. With the predicted result of entity extraction model, add `predict_entity` in the input data. For training data, regard the gold entities as `predict_entity`. Besides, in the data folder, you should add `rel.json` and `entity.json` to specify the entity type and relation type of the dataset.
```
{
    "text": "航天机电控股股东为上海航天技术研究院，实际控制人为中国航天科技集团有限公司。",
    "spo_list": [
        {
            "predicate": "投资",
            "subject_type": "公司企业",
            "object_type": "公司企业",
            "subject": "上海航天技术研究院",
            "object": "航天机电"
        },
        {
            "predicate": "投资",
            "subject_type": "公司企业",
            "object_type": "公司企业",
            "subject": "中国航天科技集团有限公司",
            "object": "航天机电"
        }
    ],
    "predict_entity": [
        [2, 5, "公司企业"],
        [11, 19, "公司企业"],
        [27, 38, "公司企业"]
    ]
}
```
Use `run_relation.py` to train a relation model. Predicted results will be saved in `results/{dataset_name}/result.json`.
```
python Run_addMarker.py \
    --dataset finance_add_predictNer \
    --batch_size 16 \
    --h_bar 0.5 \
    --t_bar 0.5
```

Arguments:
* `--dataset`: the name of the dataset directory.
* `--batch_size`: the batch size when training the model.
* `--h_bar`: the threshold of the probability to predict a head entity. 
* `--t_bar`: the threshold of the probability to predict a tail entity. 

## Support or Contact
FR3E is developed at SCTS&CGCL Lab (http://grid.hust.edu.cn/) by Weihao Wang, Mengfan Li, Hong Huang and Xuanhua Shi. For any questions, please contact Weihao Wang (whwang2020@hust.edu.cn), Mengfan Li (mengfanli1024@gmail.com), Hong Huang (honghuang@hust.edu.cn) and Xuanhua Shi (xhshi@hust.edu.cn).
