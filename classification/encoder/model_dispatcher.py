import models
from transformers import AutoTokenizer


MODEL_DISPATCHER = {
    'BERT': {
        'model_class': models.BERT,
        'tokenizer': AutoTokenizer,
        'pretrained_name': 'bert-base-uncased',
    },
    'TINY_BERT': {
        'model': models.TINY_BERT,
        'tokenizer': AutoTokenizer,
        'pretrained_name': 'prajjwal1/bert-tiny',
    },
    'RoBerta': {
        'model_class': models.ROB,
        'tokenizer': AutoTokenizer,
        'pretrained_name': 'roberta-base',
    }
}
