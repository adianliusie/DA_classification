from transformers import BertTokenizerFast, BertConfig, BertModel, ElectraModel
from transformers import RobertaTokenizerFast, RobertaModel
from transformers import BigBirdTokenizer, BigBirdModel
from transformers import LongformerTokenizerFast, LongformerModel 

from transformers import BartTokenizerFast, BartForConditionalGeneration
from transformers import LEDTokenizerFast, LEDForConditionalGeneration, LEDConfig
from transformers.models.led.modeling_led import LEDDecoder

def get_tokenizer(system):
    ### transformer encoder systems
    if system in ['bert', 'electra', 'bert_rand']: tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    elif system == 'bert_cased': tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    elif system == 'roberta':    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    elif system == 'big_bird':   tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
    elif system == 'longformer': tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")
    elif system == 'bart':              tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
    elif 'led' in system: tokenizer = LEDTokenizerFast.from_pretrained("allenai/led-base-16384")
  
    else: raise ValueError("invalid transfomer system provided")
    return tokenizer

def get_transformer(system):
    if   system ==       'bert': transformer = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
    elif system ==    'electra': transformer = ElectraModel.from_pretrained('google/electra-base-discriminator', return_dict=True)
    elif system ==  'bert_rand': transformer = BertModel(BertConfig(return_dict=True))
    elif system == 'bert_cased': transformer = BertModel.from_pretrained('bert-base-cased', return_dict=True)
    elif system ==    'roberta': transformer = RobertaModel.from_pretrained('roberta-base', return_dict=True)
    elif system ==   'big_bird': transformer = BigBirdModel.from_pretrained('google/bigbird-roberta-base', return_dict=True)
    elif system == 'longformer': transformer = LongformerModel.from_pretrained("allenai/longformer-base-4096", return_dict=True)
    elif system ==       'bart': transformer = BartForConditionalGeneration.from_pretrained('facebook/bart-base',return_dict=True)
    elif 'led' in system:        transformer = get_led_transformer(system)
    return transformer

def get_led_transformer(system):
    "handler that creates all LED models"    
    model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384", return_dict=True)
    enc_pos, dec_pos = model.led.encoder.embed_positions, model.led.decoder.embed_positions   

    if system   == 'led': pass
    elif system == 'led_rand':
        model = LEDForConditionalGeneration(model.config)    
    elif system == 'led_dec_rand':
        model.led.decoder = LEDDecoder(model.config) 
    elif 'led_simple_' in system:
        n = int(system[-1])
        model.config.decoder_layers = n
        model.led.decoder.layers = model.led.decoder.layers[:n]
        if 'led_simple_rand_' in system:
            LEDDecoder(model.config)
    
    else:
        raise ValueError("invalid LED setting provided")

    #All models will keep the learned pos embeddings, whis can be reset in the following wrapper class
    model.led.encoder.embed_positions, model.led.decoder.embed_positions = enc_pos, dec_pos

    #routing transformer.model to transformer.led to standardise naming conventions
    model.__dict__['_modules']['model'] = model.__dict__['_modules']['led']
    return model