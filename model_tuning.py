import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel


class PromptEncoder(nn.Module):
    def __init__(self, num_prompt_tokens, hidden_size, word_embedding):
        super().__init__()
        self.num_prompt_tokens = num_prompt_tokens
        self.prompt_embeddings = nn.Embedding(num_prompt_tokens, hidden_size)
        
        init_prompt_value = word_embedding[:num_prompt_tokens].clone().detach()
        self.prompt_embeddings.weight = nn.Parameter(init_prompt_value)
        
        
    def forward(self, prompts):
        return self.prompt_embeddings(prompts)
        
        
        

class PromptTuningGPT2(nn.Module):
    def __init__(self, gpt2_model, num_prompt_tokens=10):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model)
        
        for param in self.gpt2.parameters():
            param.requires_grad = False
            
        self.num_prompt_tokens = num_prompt_tokens
        self.prompt_encoder = PromptEncoder(num_prompt_tokens, self.gpt2.config.hidden_size, self.gpt2.transformer.wte.weight)
        
        
    def save_learned_prompts(self, path):
        torch.save(self.prompt_encoder.prompt_embeddings.weight.data, path)
        
    def load_learned_prompts(self, path):
        self.prompt_encoder.prompt_embeddings.weight.data = torch.load(path)


    def forward(self, input_ids, attention_mask, labels):
        prompt_tokens = torch.arange(self.num_prompt_tokens).unsqueeze(0).to(input_ids.device)
        prompt_tokens = prompt_tokens.repeat(input_ids.size(0), 1)
        
        prompt_embeddings = self.prompt_encoder(prompt_tokens)
        # print(self.gpt2.transformer.wte.shape)
        input_embeddings = self.gpt2.transformer.wte(input_ids)
        
        input_embeddings = torch.cat(
            [
                prompt_embeddings,
                input_embeddings
            ],
            dim=1
        )
        
        attention_mask = torch.cat(
            [
                torch.ones_like(prompt_tokens).to(input_ids.device),
                attention_mask
            ],
            dim=1
        )

        labels = torch.cat(
            [
                torch.ones_like(prompt_tokens).to(input_ids.device) * -100,
                labels
            ],
            dim=1
        )
        # print(labels)
        
        return self.gpt2(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            labels=labels
        )



        