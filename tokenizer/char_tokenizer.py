from pathlib import Path
import json


class CharTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.id_to_token = {
            token_id: token
            for token, token_id in vocab.items()
        }

        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"

    @classmethod
    def train(cls, text):
        special_tokens = [
            "<PAD>",
            "<UNK>",
        ]

        characters = sorted(set(text))
        tokens = special_tokens + characters

        vocab = {
            token: token_id
            for token_id, token in enumerate(tokens)
        }

        return cls(vocab)

    def encode(self, text):
        unk_id = self.vocab[self.unk_token]

        return [
            self.vocab.get(character, unk_id)
            for character in text
        ]

    def decode(self, token_ids):
        return "".join(
            self.id_to_token.get(
                token_id,
                self.unk_token
            )
            for token_id in token_ids
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                self.vocab,
                file,
                ensure_ascii=False,
                indent=2
            )

    @classmethod
    def load(cls, path):
        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:
            vocab = json.load(file)

        return cls(vocab)


def main():
    dataset_path = Path(
        "data/raw/sample.txt"
    )

    text = dataset_path.read_text(
        encoding="utf-8"
    )

    print("Training tokenizer...")
    print(
        "Characters in dataset:",
        len(text)
    )

    tokenizer = CharTokenizer.train(text)

    print(
        "Vocabulary size:",
        len(tokenizer.vocab)
    )

    sample = "PromptForge"

    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)

    print("\nSample:")
    print(sample)

    print("\nEncoded:")
    print(encoded)

    print("\nDecoded:")
    print(decoded)

    tokenizer.save(
        "tokenizer/vocab.json"
    )

    print(
        "\nTokenizer saved to:"
    )
    print(
        "tokenizer/vocab.json"
    )


if __name__ == "__main__":
    main()
