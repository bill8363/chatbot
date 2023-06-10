from transformers import T5ForConditionalGeneration as t5FCG
from transformers.models.t5.configuration_t5 import T5Config
from typing import Optional, Tuple, Union, List, Callable






class T5ForConditionalGeneration(t5FCG):
    
    def __init__(self, config: T5Config):
        super().__init__(config)
        
        
    def preprocess(self,text):
        text = text.replace("\n", "\\n").replace("\t", "\\t")
        return text

    def postprocess(self,text):
        return text.replace("\\n", "\n").replace("\\t", "\t").replace('%20','  ')
    
    
    def get_response(self,tokenizer,text, sample=True, top_p=0.9, temperature=0.7,max_length=1024,no_repeat_ngram_size=12,num_beams=1, length_penalty=0.6):
        base_info = ""
        text=base_info+text
        text = self.preprocess(text)
        
        
        encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(self.device) 
        if not sample:
          out = self.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=max_length, num_beams=num_beams, length_penalty=length_penalty,do_sample=False)
        else:
          out = self.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=max_length, do_sample=True, top_p=top_p, temperature=temperature, no_repeat_ngram_size=no_repeat_ngram_size)
        out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
        return self.postprocess(out_text[0])
    
    
    def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, sample=True, top_p=0.9, temperature=0.7,max_length=2048,no_repeat_ngram_size=12,num_beams=1, length_penalty=0.6):
        
        
        history = history or []
        if len(history) > 5:
            history = history[-5:]

        context = "\n".join([f"用户：{input_text}\n小元：{answer_text}" for input_text, answer_text in history])
        #print(context)

        input_text = context + "\n用户：" + query + "\n小元："
        input_text = input_text.strip()
        response = self.get_response(tokenizer,input_text,sample=sample, top_p=top_p, temperature=temperature,max_length=max_length,no_repeat_ngram_size=no_repeat_ngram_size,num_beams=num_beams, length_penalty=length_penalty)

        history.append((query, response))
        return response,history
