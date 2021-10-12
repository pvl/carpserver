import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer


class Parapharaser:
    """ Paraphrases using peagasus. Used for softening. """

    def __init__(self, device="cuda"):
        self.model_name = 'tuner007/pegasus_paraphrase'
        self.device = device
        self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name).half().to(self.device)


    def get(self, input_text, num_return_sequences, num_beams=3):
        batch = self.tokenizer([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(self.device)
        translated = self.model.generate(**batch, max_length=60, num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text