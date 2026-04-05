import torch
from torch.utils.data import Dataset


class SVGSeq2SeqDataset(Dataset):
    """PyTorch Dataset that tokenizes prompt/SVG pairs for seq2seq training.

    Args:
        df (pd.DataFrame): DataFrame containing input and target text columns.
        tokenizer: Seq2seq tokenizer used for both inputs and targets.
        max_input_length (int, optional): Maximum input token length. Defaults to ``64``.
        max_target_length (int, optional): Maximum target token length. Defaults to ``512``.
        input_col (str, optional): Column containing the source text. Defaults to ``"input_text"``.
        target_col (str, optional): Column containing the target SVG text. Defaults to
            ``"target_text"``.
    """

    def __init__(
        self,
        df,
        tokenizer,
        max_input_length: int = 64,
        max_target_length: int = 512,
        input_col: str = "input_text",
        target_col: str = "target_text",
    ):
        """Store tokenization settings and normalized training rows.

        Args:
            df (pd.DataFrame): DataFrame containing one training example per row.
            tokenizer: Seq2seq tokenizer used to encode source and target text.
            max_input_length (int, optional): Maximum encoded length for source text.
            max_target_length (int, optional): Maximum encoded length for target text.
            input_col (str, optional): Source-text column name.
            target_col (str, optional): Target-text column name.

        Returns:
            None: Dataset state is stored on ``self`` for later indexing.
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.input_col = input_col
        self.target_col = target_col

    def __len__(self):
        """Return the number of examples in the dataset.

        Args:
            None.

        Returns:
            int: Number of rows stored in ``self.df``.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """Tokenize one training example for seq2seq supervision.

        Args:
            idx (int): Row index in ``self.df``.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing ``input_ids``, ``attention_mask``, and
                ``labels``. Padding tokens in ``labels`` are replaced with ``-100`` so they are
                ignored by the loss function.
        """
        input_text = str(self.df.loc[idx, self.input_col])
        target_text = str(self.df.loc[idx, self.target_col])
        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_enc = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = target_enc["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": labels,
        }
