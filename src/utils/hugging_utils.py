from transformers import BertTokenizerFast, BertConfig, BertModel, ElectraModel
from transformers import RobertaTokenizerFast, RobertaModel
from transformers import BigBirdTokenizer, BigBirdModel
from transformers import LongformerTokenizerFast, LongformerModel 

from transformers import BartTokenizerFast, BartForConditionalGeneration
from transformers import LEDTokenizerFast, LEDForConditionalGeneration 

def get_tokenizer(system):
    ### transformer encoder systems
    if system in ['bert', 'electra', 'bert-rand']: tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    elif system == 'bert_cased': tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    elif system == 'roberta':    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    elif system == 'big_bird':   tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
    elif system == 'longformer': tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")
    ### seq2seq systems
    elif system == 'bart':       tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
    elif system == 'led':        tokenizer = LEDTokenizerFast.from_pretrained("allenai/led-base-16384")
    else: raise Exception
    return tokenizer

def get_transformer(system):
    if   system ==       'bert': transformer = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
    elif system ==    'electra': transformer = ElectraModel.from_pretrained('google/electra-base-discriminator', return_dict=True)
    elif system ==  'bert-rand': transformer = BertModel(BertConfig(return_dict=True))
    elif system == 'bert_cased': transformer = BertModel.from_pretrained('bert-base-cased', return_dict=True)
    elif system ==    'roberta': transformer = RobertaModel.from_pretrained('roberta-base', return_dict=True)
    elif system ==   'big_bird': transformer = BigBirdModel.from_pretrained('google/bigbird-roberta-base', return_dict=True)
    elif system == 'longformer': transformer = LongformerModel.from_pretrained("allenai/longformer-base-4096", return_dict=True)
    elif system == 'bart': transformer = BartForConditionalGeneration.from_pretrained('facebook/bart-base', return_dict=True)
    elif system ==  'led': 
        transformer = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384", return_dict=True)
        #routing transformer.model to transformer.led to standardise naming conventions
        transformer.__dict__['_modules']['model'] = transformer.__dict__['_modules']['led']
    return transformer
