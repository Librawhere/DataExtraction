import torch.nn as nn
from transformers import GPTNeoXForCausalLM


class Pythia(nn.Module):
    def __init__(self):
        super(Pythia, self).__init__()

        self.transformer = GPTNeoXForCausalLM.from_pretrained(
            pretrained_model_name_or_path="EleutherAI/pythia-70m-deduped",
            revision="step3000",
            cache_dir="./pythia-70m-deduped/step3000",
        )

    def forward(self, x, y):
        outputs = self.transformer(input_ids=x, labels=y)
        return outputs

    def generate(self, input_ids, attention_mask, max_length, pad_token_id):
        # greedy-decoding
        # output = self.transformer.generate(input_ids=input_ids,
        #                                    max_length=max_length,
        #                                    no_repeat_ngram_size=2,
        #                                    early_stopping=False,
        #                                    attention_mask=attention_mask,
        #                                    pad_token_id=pad_token_id)
        # beam search
        output = self.transformer.generate(input_ids=input_ids,
                                           max_length=max_length,
                                           num_beams=5,
                                           no_repeat_ngram_size=2,
                                           num_return_sequences=5,
                                           early_stopping=False,
                                           attention_mask=attention_mask,
                                           pad_token_id=pad_token_id)

        return output


if __name__ == '__main__':
    sentences = ['the sun', 'the sea', 'the river']
    sentences = " <|endoftext|> ".join(sentences)

    print(sentences)













