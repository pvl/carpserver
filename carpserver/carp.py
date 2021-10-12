import sys
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path
import transformers
from transformers import AutoModel, AutoTokenizer
from .paraphrase import Parapharaser


LATENT_DIM = 2048
USE_CUDA = True
USE_HALF = True
N_CTX = 512
MODEL_PATH = "roberta-large"

config = transformers.RobertaConfig()

extract_fns = {'EleutherAI/gpt-neo-1.3B' :
                (lambda out : out['hidden_states'][-1]),
                'EleutherAI/gpt-neo-2.7B' :
                (lambda out : out['hidden_states'][-1]),
                'roberta-large' :
                (lambda out : out[0]),
                'roberta-base' :
                (lambda out : out[0]),
                'microsoft/deberta-v2-xlarge' :
                (lambda out : out[0])}

d_models = {'EleutherAI/gpt-neo-1.3B' : 2048,
            'EleutherAI/gpt-neo-2.7B' : 2560,
            'roberta-large' : 1024,
            'roberta-base' : 768,
            'microsoft/deberta-v2-xlarge' : 1536}


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = AutoModel.from_pretrained(MODEL_PATH)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.d_model = d_models[MODEL_PATH]

        # Add cls token to model and tokenizer
        self.tokenizer.add_tokens(['[quote]'])
        self.model.resize_token_embeddings(len(self.tokenizer))

    def tok(self, string_batch):
        return self.tokenizer(string_batch,
                return_tensors = 'pt',
                padding = True).to('cuda')

    def forward(self, x, mask = None, tokenize = False, mask_sum = True):
        if tokenize:
            x = self.tok(x)
            mask = x['attention_mask']
            x = x['input_ids']

        out = self.model(x, mask, output_hidden_states = True, return_dict = True)

        # out is a tuple of (model output, tuple)
        # the second tuple is all layers
        # in this second tuple, last elem is model output
        # we take second last hidden -> third last layer
        # size is always [batch, seq, 1536]

        hidden = out[0]
        #layers = out[-1]
        #hidden = layers[-2]

        # Mask out pad tokens embeddings
        if mask_sum:
            emb_mask = mask.unsqueeze(2).repeat(1, 1, self.d_model)
            hidden = hidden * emb_mask

        y = hidden.sum(1)
        y = F.normalize(y)

        return y # Sum along sequence


class ContrastiveModel(nn.Module):
    def __init__(self, encA, encB):
        super().__init__()

        self.encA = encA
        self.encB = encB

        self.projA = nn.Linear(self.encA.d_model, LATENT_DIM, bias = False)
        self.projB = nn.Linear(self.encB.d_model, LATENT_DIM, bias = False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.clamp_min = math.log(1/100)
        self.clamp_max = math.log(100)

    def clamp(self):
        with torch.no_grad():
            self.logit_scale.clamp(self.clamp_min, self.clamp_max)

    def encodeX(self, x, masks = None):
        x = self.encA(x, masks)
        return self.projA(x)

    def encodeY(self, y, masks = None):
        y = self.encB(y, masks)
        return self.projB(y)

    # Calculate contrastive loss between embedding groups
    # x, y are assumed encoding/embeddings here
    def cLoss(self, x, y):
        n = x.shape[0]
        # normalize
        x = F.normalize(x)
        y = F.normalize(y)

        logits = x @ y.T * self.logit_scale.exp()
        labels = torch.arange(n, device ='cuda')

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        acc_i = (torch.argmax(logits, dim = 1) == labels).sum()
        acc_t = (torch.argmax(logits, dim = 0) == labels).sum()

        return (loss_i + loss_t) / 2, (acc_i + acc_t) / n / 2

    def getLogits(self, x, y):
        x = self.encodeX(*x)
        y = self.encodeY(*y)

        x = F.normalize(x)
        y = F.normalize(y)

        logits = x @ y.T * self.logit_scale.exp()
        return logits

    def forward(self, x, y):
        return self.getLogits(x, y)


def report_logits(logits):
    logits /= 2.7441
    print((logits[0]).cpu().tolist())
    conf = logits.softmax(1)

    for i, row in enumerate(conf):
        for j, col in enumerate(row):
            print(str(i) + "-" + str(j) + ": " + str(round(col.item(), 2)))


class TextReviewer:
    def __init__(self):
        self.carp_model_path = Path(__file__).parent.parent / "models" / "CARP_L.pt"
        if not self.carp_model_path.exists():
            print("Model file CARP_L.pt not found in models folder")
            print("Download the model from https://the-eye.eu/public/AI/CARP_L.pt")
            print("And save it in the models folder")
            sys.exit(0)

        self.pm = Parapharaser()
        self.model = ContrastiveModel(TextEncoder(), TextEncoder())
        self.model.load_state_dict(torch.load(self.carp_model_path))
        if USE_HALF: self.model.half()
        if USE_CUDA: self.model.cuda()

    def tok(self, string_batch):
        for i, _ in enumerate(string_batch):
            if len(string_batch[i]) > N_CTX:
                string_batch[i] = string_batch[i][-N_CTX:]

        return self.model.encA.tok(string_batch)

    def get_batch_tokens(self, dataset, inds):
        batch = [dataset[ind] for ind in inds]
        pass_batch = [pair[0] for pair in batch]
        rev_batch = [pair[1] for pair in batch]

        pass_tokens = self.tok(pass_batch)
        rev_tokens = self.tok(rev_batch)
        pass_masks = pass_tokens['attention_mask']
        rev_masks = rev_tokens['attention_mask']
        pass_tokens = pass_tokens['input_ids']
        rev_tokens = rev_tokens['input_ids']

        return pass_tokens, pass_masks, rev_tokens, rev_masks

    def get_passrev_logits(self, passages, reviews):
        """ Compute the logits of the passage against the reviews """
        pass_tokens = self.tok(passages)
        rev_tokens = self.tok(reviews)
        pass_masks = pass_tokens['attention_mask']
        rev_masks = rev_tokens['attention_mask']
        pass_tokens = pass_tokens['input_ids']
        rev_tokens = rev_tokens['input_ids']

        with torch.no_grad():
            logits = self.model.getLogits([pass_tokens, pass_masks],
                            [rev_tokens, rev_masks]).type(dtype=torch.float32)
        return logits

    def compute_softened_logits(self, passages, reviews1, reviews2, pairs=True):

        logits1 = torch.sum(self.get_passrev_logits(passages, reviews1), dim=-1).unsqueeze(0)/float(len(reviews1))
        if pairs:
            logits2 = torch.sum(self.get_passrev_logits(passages, reviews2), dim=-1).unsqueeze(0)/float(len(reviews2))
            return torch.cat([logits1, logits2], dim=-1)
        else:
            return logits1


    #Lots of options to play with here that dictate how the paraphrases are generated.
    #Future work is needed
    def compute_logit(self, passages, reviews, soften=True,
                            top_k=False, k = 3,
                            ret = False, pairs=True):
        #Softens the classifiers by using paraphrasing.
        if soften:
            if pairs:
                review1_paraphrases = list(set(self.pm.get(reviews[0], num_return_sequences=3) + [reviews[0]]))
                review2_paraphrases = list(set(self.pm.get(reviews[1], num_return_sequences=3) + [reviews[1]]))
                print(review1_paraphrases)
                print(review2_paraphrases)

                review1_contextual = list(map(lambda x: "[quote] " + x, review1_paraphrases))
                review2_contextual = list(map(lambda x: "[quote] " + x, review2_paraphrases))


                softened_logits = self.compute_softened_logits(passages, review1_contextual + review1_paraphrases, review2_contextual + review2_paraphrases)
                report_logits(softened_logits)
                if ret:
                    return softened_logits
            else:
                review_paraphrases = list(set(self.pm.get(reviews, num_return_sequences=3) + [reviews]))
                #print(review_paraphrases)

                review_contextual = list(map(lambda x: "[quote] " + x, review_paraphrases))
                softened_logits = self.compute_softened_logits(passages, review_contextual + review_paraphrases, None, pairs=False)

                #softened_logits = (softened_logits/2.7441)
                print(softened_logits.squeeze().cpu().tolist())

                if ret:
                    return softened_logits



