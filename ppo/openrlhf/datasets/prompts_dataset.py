from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none


def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None, context_key=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
    if context_key is not None: 
        return (data[context_key], prompt)
    return prompt


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        context_key = getattr(self.strategy.args, "context_key", None)    # context_key->context; input_key->prompt (after template)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.contexts = [] if (context_key is not None) else None
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, input_key, apply_chat_template, context_key)
            if context_key is not None:
                context = prompt[0]
                prompt = prompt[1]
                self.contexts.append(context)
            self.prompts.append(prompt)
        self.get_context = (context_key is not None)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        if self.get_context:
            return (self.contexts[idx], self.prompts[idx])
        return self.prompts[idx]
