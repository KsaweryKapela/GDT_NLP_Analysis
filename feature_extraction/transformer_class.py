from typing import List, Tuple
from sklearn.base import TransformerMixin, BaseEstimator
import torch
import pandas as pd

class BertTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bert_tokenizer, bert_model, max_length):
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.bert_tokenizer = bert_tokenizer
        self.bert_model = bert_model
        self.bert_model.eval()
        self.max_length = max_length
        self.embedding_func = lambda x: torch.mean(x[0], dim=1).squeeze()

    def _tokenize(self, text: str) -> Tuple[torch.tensor, torch.tensor]:
        tokenized_text = self.bert_tokenizer.encode_plus(text,
                                                         add_special_tokens=True,
                                                         return_tensors='pt',
                                                         max_length=self.max_length,
                                                         )["input_ids"].to(self.device)

        attention_mask = [1] * len(tokenized_text)

        return (
            torch.tensor(tokenized_text).unsqueeze(0).to(self.device),
            torch.tensor(attention_mask).unsqueeze(0).to(self.device),
        )

    def _tokenize_and_predict(self, text: str) -> torch.tensor:
        
        tokenized, attention_mask = self._tokenize(text)

        embeddings = self.bert_model(tokenized[0], attention_mask)
        
        return self.embedding_func(embeddings)

    def transform(self, text: List[str]):
        if isinstance(text, pd.Series):
            text = text.tolist()

        with torch.no_grad():
            return torch.stack([self._tokenize_and_predict(string) for string in text])