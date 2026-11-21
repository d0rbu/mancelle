"""
BPE Tokenizer in the style of GPT-4.

Two implementations are available:
1) HuggingFace Tokenizer that can do both training and inference but is really confusing
2) Our own RustBPE Tokenizer for training and tiktoken for efficient inference
"""

import copy
import json
import os
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise

import rustbpe
import tiktoken
import torch
from tokenizers import Regex, decoders, pre_tokenizers
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer

from errors import ParseError
from nanochat.common import get_base_dir

# Migration map: old pipe-delimited format → new Gemma-style format.
# Applied transparently when loading tokenizers trained with the old format.
_SPECIAL_TOKEN_MIGRATION = {
    "<|bos|>": "<bos>",
    "<|user_start|>": "<user_start>",
    "<|user_end|>": "<user_end>",
    "<|assistant_start|>": "<assistant_start>",
    "<|assistant_end|>": "<assistant_end>",
    "<|python_start|>": "<python_start>",
    "<|python_end|>": "<python_end>",
    "<|output_start|>": "<output_start>",
    "<|output_end|>": "<output_end>",
}


def _migrate_special_tokens(enc: tiktoken.Encoding) -> tiktoken.Encoding:
    """Re-create a tiktoken Encoding with renamed special tokens if needed."""
    old_specials: dict[str, int] = enc._special_tokens
    needs_migration = any(k in _SPECIAL_TOKEN_MIGRATION for k in old_specials)
    if not needs_migration:
        return enc
    new_specials = {
        _SPECIAL_TOKEN_MIGRATION.get(k, k): v for k, v in old_specials.items()
    }
    # NOTE: accesses tiktoken private attributes (_special_tokens, _pat_str,
    # _mergeable_ranks) to reconstruct the Encoding with renamed tokens.
    # Stable since tiktoken 0.5; pin version if this breaks.
    return tiktoken.Encoding(
        name=enc.name,
        pat_str=enc._pat_str,
        mergeable_ranks=enc._mergeable_ranks,
        special_tokens=new_specials,
    )


SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<bos>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<user_start>",  # user messages
    "<user_end>",
    "<assistant_start>",  # assistant messages
    "<assistant_end>",
    "<python_start>",  # assistant invokes python REPL tool
    "<python_end>",
    "<output_start>",  # python REPL outputs back to assistant
    "<output_end>",
]


@dataclass(frozen=True)
class LogicalTurnSpan:
    """One half-open role span in a structured conversation target stream."""

    role: str
    start: int
    end: int


@dataclass(frozen=True)
class StructuredConversationRender:
    """Authoritative, immutable rendering used by structured SFT artifacts.

    ``token_ids`` and ``assistant_mask`` include decoder BOS. ``logical_turns``
    index the BOS-free target stream, so the first non-BOS token is position 0.
    """

    token_ids: tuple[int, ...]
    assistant_mask: tuple[bool, ...]
    logical_turns: tuple[LogicalTurnSpan, ...]
    literal_tag_collisions: int


_STRUCTURED_WIRE_TAGS = (
    "<system>",
    "</system>",
    "<tool_call>",
    "</tool_call>",
    "<tool_result>",
    "</tool_result>",
)


def _canonicalize_json_value(value):
    """Recursively key-canonicalize one JSON value without changing arrays."""
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("tool JSON objects must have string keys")
        return {
            key: _canonicalize_json_value(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(
        f"tool JSON contains unsupported value type {type(value).__name__}"
    )


def _compact_json(fields):
    """Serialize fixed-order top-level fields with canonical nested mappings."""
    canonical = {
        key: _canonicalize_json_value(value)
        for key, value in fields
    }
    try:
        return json.dumps(
            canonical,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tool payload is not canonical JSON: {exc}") from exc


def _count_structured_tag_collisions(value):
    """Count literal wire tags in semantic source strings, recursively."""
    if isinstance(value, str):
        return sum(value.count(tag) for tag in _STRUCTURED_WIRE_TAGS)
    if isinstance(value, Mapping):
        return sum(
            _count_structured_tag_collisions(key)
            + _count_structured_tag_collisions(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(_count_structured_tag_collisions(item) for item in value)
    return 0

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I verified that 2 is the sweet spot for vocab size of 32K. 1 is a bit worse, 3 was worse still.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        # init from a HuggingFace pretrained tokenizer (e.g. "gpt2")
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # init from a local directory on disk (e.g. "out/tokenizer")
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # train from an iterator of text
        # Configure the HuggingFace Tokenizer
        tokenizer = HFTokenizer(
            BPE(
                byte_fallback=True,  # needed!
                unk_token=None,
                fuse_unk=False,
            )
        )
        # Normalizer: None
        tokenizer.normalizer = None
        # Pre-tokenizer: GPT-4 style
        # the regex pattern used by GPT-4 to split text into groups before BPE
        # NOTE: The pattern was changed from \p{N}{1,3} to \p{N}{1,2} because I suspect it is harmful to
        # very small models and smaller vocab sizes, because it is a little bit wasteful in the token space.
        # (but I haven't validated this! TODO)
        gpt4_split_regex = Regex(
            SPLIT_PATTERN
        )  # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    pattern=gpt4_split_regex, behavior="isolated", invert=False
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        tokenizer.post_processor = None
        # Trainer: BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0,  # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None, num_threads=None):
        # encode a single string
        # prepend/append can be either a string of a special token or a token id directly.
        # num_threads is ignored (only used by the nanochat Tokenizer for parallel encoding)
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = (
                prepend if isinstance(prepend, int) else self.encode_special(prepend)
            )
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = (
                append if isinstance(append, int) else self.encode_special(append)
            )
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        # encode a single special token via exact match
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        # Different HuggingFace models use different BOS tokens and there is little consistency
        # 1) attempt to find a <bos> token
        bos = self.encode_special("<bos>")
        # 2) if that fails, attempt to find a <|endoftext|> token (e.g. GPT-2 models)
        if bos is None:
            bos = self.encode_special("<|endoftext|>")
        # 3) if these fail, it's better to crash than to silently return None
        assert bos is not None, "Failed to find BOS token in tokenizer"
        return bos

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        # save the tokenizer to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")


class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()
        # the special tokens are inserted later in __init__, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, (
            f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        )
        tokenizer.train_from_iterator(
            text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN
        )
        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {
            name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)
        }
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,  # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens,  # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<bos>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        enc = _migrate_special_tokens(enc)
        return cls(enc, "<bos>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<bos>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text can be either a string or a list of strings

        prepend_id: int | None = None
        append_id: int | None = None
        if prepend is not None:
            prepend_id = (
                prepend if isinstance(prepend, int) else self.encode_special(prepend)
            )
        if append is not None:
            append_id = (
                append if isinstance(append, int) else self.encode_special(append)
            )

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend_id is not None:
                ids.insert(0, prepend_id)  # TODO: slightly inefficient here? :( hmm
            if append_id is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend_id is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)  # TODO: same
            if append_id is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        # save the encoding object to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).
        Returns:
        - ids: list[int] is a list of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.
        """
        # ids, masks that we will return and a helper function to help build them up.
        ids, mask = [], []

        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        if conversation["messages"][0]["role"] == "system":
            # some conversation surgery is necessary here for now...
            conversation = copy.deepcopy(conversation)  # avoid mutating the original
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", (
                "System message must be followed by a user message"
            )
            messages[1]["content"] = (
                messages[0]["content"] + "\n\n" + messages[1]["content"]
            )
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all the special tokens we need
        bos = self.get_bos_token_id()
        user_start, user_end = (
            self.encode_special("<user_start>"),
            self.encode_special("<user_end>"),
        )
        assistant_start, assistant_end = (
            self.encode_special("<assistant_start>"),
            self.encode_special("<assistant_end>"),
        )
        python_start, python_end = (
            self.encode_special("<python_start>"),
            self.encode_special("<python_end>"),
        )
        output_start, output_end = (
            self.encode_special("<output_start>"),
            self.encode_special("<output_end>"),
        )

        # now we can tokenize the conversation
        add_tokens(bos, 0)
        for i, message in enumerate(messages):
            # some sanity checking here around assumptions, to prevent footguns
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, (
                f"Message {i} is from {message['role']} but should be from {must_be_from}"
            )

            # content can be either a simple string or a list of parts (e.g. containing tool calls)
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), (
                    "User messages are simply expected to be strings"
                )
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # python tool call => add the tokens inside <python_start> and <python_end>
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python output => add the tokens inside <output_start> and <output_end>
                            # none of these tokens are supervised because the tokens come from Python at test time
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # truncate to max_tokens tokens MAX (helps prevent OOMs)
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def render_structured_conversation(self, conversation):
        """Render one complete structured SFT conversation without truncation.

        This is deliberately additive: :meth:`render_conversation` remains the
        legacy alternating-message renderer, including its 2,048-token cap.
        Structured callers receive authoritative token IDs, a BOS-aligned
        assistant mask, and BOS-free logical role spans. Decoded text is never
        used to recover token IDs or roles.
        """
        if not isinstance(conversation, Mapping):
            raise ValueError(
                "structured conversation must be a mapping containing 'messages'"
            )
        if "messages" not in conversation:
            raise ValueError(
                "structured conversation requires the top-level 'messages' field"
            )
        # Task rows may carry inert provenance (for example SpellingBee's
        # ``letters`` and ``subject``). Logical content still comes exclusively
        # from ``messages``. Reject alternate semantic inputs that would
        # otherwise look accepted while being silently discarded.
        forbidden_top_level = {
            "assistant_loss_mask",
            "full_nanochat_text",
            "pre_rendered_text",
            "prompt",
            "python_tools",
            "rendered_text",
            "response",
            "text",
            "tool_choice",
            "tools",
        }
        present_forbidden = sorted(forbidden_top_level.intersection(conversation))
        if present_forbidden:
            raise ValueError(
                "structured conversation rejects alternate top-level semantic fields; "
                f"got {present_forbidden}"
            )
        source_messages = conversation["messages"]
        if not isinstance(source_messages, (list, tuple)) or not source_messages:
            raise ValueError("structured conversation messages must be a non-empty sequence")
        messages = list(source_messages)

        system_content = None
        if isinstance(messages[0], Mapping) and messages[0].get("role") == "system":
            system_message = messages.pop(0)
            if set(system_message) != {"role", "content"}:
                raise ValueError("system messages require exactly role and content")
            system_content = system_message["content"]
            if not isinstance(system_content, str):
                raise ValueError("system message content must be a string")
        if not messages:
            raise ValueError("a system message must be followed by a user message")

        bos = self.get_bos_token_id()
        user_start, user_end = (
            self.encode_special("<user_start>"),
            self.encode_special("<user_end>"),
        )
        assistant_start, assistant_end = (
            self.encode_special("<assistant_start>"),
            self.encode_special("<assistant_end>"),
        )
        python_start, python_end = (
            self.encode_special("<python_start>"),
            self.encode_special("<python_end>"),
        )
        output_start, output_end = (
            self.encode_special("<output_start>"),
            self.encode_special("<output_end>"),
        )

        ids = [bos]
        mask = [False]
        spans = []
        collision_count = _count_structured_tag_collisions(source_messages)

        def add_tokens(token_ids, supervised, role):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            else:
                token_ids = list(token_ids)
            if not token_ids:
                return
            start = len(ids) - 1
            ids.extend(token_ids)
            mask.extend([bool(supervised)] * len(token_ids))
            end = len(ids) - 1
            if spans and spans[-1].role == role and spans[-1].end == start:
                previous = spans[-1]
                spans[-1] = LogicalTurnSpan(role=role, start=previous.start, end=end)
            else:
                spans.append(LogicalTurnSpan(role=role, start=start, end=end))

        def require_message(message, role, fields):
            if not isinstance(message, Mapping):
                raise ValueError(
                    f"structured {role} message must be a mapping, got "
                    f"{type(message).__name__}"
                )
            if message.get("role") != role or set(message) != set(fields):
                raise ValueError(
                    f"structured {role} messages require exactly {sorted(fields)}"
                )

        expect = "user"
        outstanding_tool = None
        for message_index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise ValueError(
                    f"message {message_index} must be a mapping, got "
                    f"{type(message).__name__}"
                )
            role = message.get("role")
            if role == "system":
                raise ValueError("system messages are allowed only at position zero")

            if role == "user":
                if expect != "user":
                    raise ValueError(
                        f"message {message_index} is user while structured grammar expects {expect}"
                    )
                require_message(message, "user", ("role", "content"))
                content = message["content"]
                if not isinstance(content, str):
                    raise ValueError("user message content must be a string")
                if system_content is not None:
                    content = f"<system>{system_content}</system>\n\n{content}"
                    system_content = None
                add_tokens(user_start, False, "user")
                add_tokens(self.encode(content), False, "user")
                add_tokens(user_end, False, "user")
                expect = "assistant"
                continue

            if role == "tool":
                if expect != "tool" or outstanding_tool is None:
                    raise ValueError(
                        f"message {message_index} is an orphan tool result"
                    )
                require_message(message, "tool", ("role", "name", "content"))
                name = message["name"]
                content = message["content"]
                if not isinstance(name, str) or not name:
                    raise ValueError("tool result name must be a non-empty string")
                if not isinstance(content, str):
                    raise ValueError("tool result content must be a string")
                if name != outstanding_tool:
                    raise ValueError(
                        f"tool result name {name!r} does not match outstanding call "
                        f"{outstanding_tool!r}"
                    )
                if name == "python":
                    add_tokens(output_start, False, "tool")
                    add_tokens(self.encode(content), False, "tool")
                    add_tokens(output_end, False, "tool")
                else:
                    payload = _compact_json((("name", name), ("content", content)))
                    add_tokens(
                        self.encode(f"<tool_result>{payload}</tool_result>"),
                        False,
                        "tool",
                    )
                outstanding_tool = None
                expect = "assistant"
                continue

            if role != "assistant":
                raise ValueError(f"unsupported structured message role {role!r}")
            if expect != "assistant":
                raise ValueError(
                    f"message {message_index} is assistant while structured grammar expects {expect}"
                )
            require_message(message, "assistant", ("role", "content"))
            content = message["content"]
            add_tokens(assistant_start, False, "assistant")
            ends_in_generic_call = False
            if isinstance(content, str):
                add_tokens(self.encode(content), True, "assistant")
            elif isinstance(content, (list, tuple)):
                if not content:
                    raise ValueError("assistant multipart content must not be empty")
                native_call_open = False
                needs_native_continuation = False
                for part_index, part in enumerate(content):
                    if not isinstance(part, Mapping) or not isinstance(part.get("type"), str):
                        raise ValueError(
                            f"assistant part {part_index} must be a mapping with a string type"
                        )
                    part_type = part["type"]
                    if part_type in {"text", "python", "python_output"}:
                        if set(part) != {"type", "text"} or not isinstance(part["text"], str):
                            raise ValueError(
                                f"assistant {part_type} parts require exactly type and string text"
                            )
                        text = part["text"]
                        if part_type == "text":
                            if native_call_open:
                                raise ValueError(
                                    "native python output must immediately follow its call"
                                )
                            add_tokens(self.encode(text), True, "assistant")
                            if text:
                                needs_native_continuation = False
                        elif part_type == "python":
                            if native_call_open:
                                raise ValueError("parallel native python calls are unsupported")
                            native_call_open = True
                            add_tokens(python_start, True, "assistant")
                            add_tokens(self.encode(text), True, "assistant")
                            add_tokens(python_end, True, "assistant")
                        else:
                            if not native_call_open:
                                raise ValueError("orphan native python_output part")
                            native_call_open = False
                            needs_native_continuation = True
                            add_tokens(output_start, False, "tool")
                            add_tokens(self.encode(text), False, "tool")
                            add_tokens(output_end, False, "tool")
                        continue

                    if part_type != "tool_call":
                        raise ValueError(f"unknown assistant part type {part_type!r}")
                    if set(part) != {"type", "name", "arguments"}:
                        raise ValueError(
                            "generic tool_call parts require exactly type, name, and arguments"
                        )
                    if native_call_open:
                        raise ValueError("a generic tool call cannot overlap a native python call")
                    if part_index != len(content) - 1:
                        raise ValueError("a generic tool call must end its assistant portion")
                    name = part["name"]
                    if not isinstance(name, str) or not name:
                        raise ValueError("tool call name must be a non-empty string")
                    arguments = part["arguments"]
                    if name == "python":
                        python_text = (
                            arguments
                            if isinstance(arguments, str)
                            else json.dumps(
                                _canonicalize_json_value(arguments),
                                ensure_ascii=False,
                                allow_nan=False,
                                separators=(",", ":"),
                            )
                        )
                        add_tokens(python_start, True, "assistant")
                        add_tokens(self.encode(python_text), True, "assistant")
                        add_tokens(python_end, True, "assistant")
                    else:
                        payload = _compact_json(
                            (("name", name), ("arguments", arguments))
                        )
                        add_tokens(
                            self.encode(f"<tool_call>{payload}</tool_call>"),
                            True,
                            "assistant",
                        )
                    outstanding_tool = name
                    ends_in_generic_call = True
                if native_call_open:
                    raise ValueError("assistant native python call has no output")
                if needs_native_continuation:
                    raise ValueError(
                        "assistant native call/result sequence has no final continuation"
                    )
            else:
                raise ValueError(
                    "assistant message content must be a string or multipart sequence"
                )
            add_tokens(assistant_end, True, "assistant")
            expect = "tool" if ends_in_generic_call else "user"

        if system_content is not None:
            raise ValueError("a system message must be followed by a user message")
        if expect != "user" or outstanding_tool is not None:
            raise ValueError(f"structured conversation ends with an incomplete {expect} phase")
        if len(mask) != len(ids):
            raise AssertionError("structured renderer produced a misaligned token mask")
        if not spans or spans[0].start != 0 or spans[-1].end != len(ids) - 1:
            raise AssertionError("structured renderer spans do not cover the target stream")
        for left, right in pairwise(spans):
            if left.end != right.start:
                raise AssertionError("structured renderer spans contain a gap or overlap")
        return StructuredConversationRender(
            token_ids=tuple(ids),
            assistant_mask=tuple(mask),
            logical_turns=tuple(spans),
            literal_tag_collisions=collision_count,
        )

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        GRAY = "\033[90m"
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return "|".join(tokens)

    def render_for_completion(self, conversation):
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.
        """
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        conversation = copy.deepcopy(conversation)  # avoid mutating the original
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", (
            "Last message must be from the Assistant"
        )
        messages.pop()  # remove the last message (of the Assistant) inplace

        # Now tokenize the conversation
        ids, mask = self.render_conversation(conversation)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<assistant_start>")
        ids.append(assistant_start)
        return ids


class TransformersTokenizer:
    """Wrapper around HuggingFace transformers AutoTokenizer for pretrained tokenizer reuse.

    Conforms to the same interface as RustBPETokenizer so it can be used as a
    drop-in replacement throughout nanochat (dataloader, training, evaluation).
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Resolve BOS token
        if tokenizer.bos_token_id is not None:
            self._bos_token_id = tokenizer.bos_token_id
        elif tokenizer.eos_token_id is not None:
            # Fallback: some tokenizers only define EOS (e.g. GPT-2 uses <|endoftext|> for both)
            self._bos_token_id = tokenizer.eos_token_id
        else:
            raise ValueError(
                "Tokenizer has no BOS or EOS token to use as document delimiter"
            )

    @classmethod
    def from_pretrained(cls, model_name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        return cls(tokenizer)

    def get_vocab_size(self):
        return len(self.tokenizer)

    def get_special_tokens(self):
        return set(self.tokenizer.all_special_tokens)

    def id_to_token(self, id):
        return self.tokenizer.convert_ids_to_tokens(id)

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        """Encode a single special token. Maps nanochat special token names to this tokenizer's tokens."""
        # Map nanochat's canonical <bos> to whatever this tokenizer uses
        if text == "<bos>":
            return self._bos_token_id
        # Try direct lookup in the tokenizer's vocabulary
        token_id = self.tokenizer.convert_tokens_to_ids(text)
        if token_id != self.tokenizer.unk_token_id:
            return token_id
        raise ParseError(f"Special token {text!r} not found in tokenizer vocabulary")

    def get_bos_token_id(self):
        return self._bos_token_id

    def _encode_one(self, text, prepend=None, append=None, num_threads=None):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if prepend is not None:
            prepend_id = (
                prepend if isinstance(prepend, int) else self.encode_special(prepend)
            )
            ids.insert(0, prepend_id)
        if append is not None:
            append_id = (
                append if isinstance(append, int) else self.encode_special(append)
            )
            ids.append(append_id)
        return ids

    def encode(self, text, prepend=None, append=None, num_threads=8):
        if isinstance(text, str):
            return self._encode_one(text, prepend=prepend, append=append)
        elif isinstance(text, list):
            return [self._encode_one(t, prepend=prepend, append=append) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        self.tokenizer.save_pretrained(tokenizer_dir)
        print(f"Saved tokenizer to {tokenizer_dir}")


# -----------------------------------------------------------------------------
# nanochat-specific convenience functions


def get_tokenizer():
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    # Auto-detect tokenizer type: transformers saves tokenizer_config.json,
    # RustBPE saves tokenizer.pkl
    if os.path.exists(os.path.join(tokenizer_dir, "tokenizer_config.json")):
        return TransformersTokenizer.from_directory(tokenizer_dir)
    return RustBPETokenizer.from_directory(tokenizer_dir)


def get_token_bytes(device="cpu"):
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), (
        f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    )
    with open(token_bytes_path, "rb") as f:
        # weights_only=True refuses arbitrary pickle opcodes; token_bytes is
        # a plain tensor so the safe-globals allowlist is sufficient.
        # Ref: Wave 3B security hardening (pickle RCE closure).
        token_bytes = torch.load(f, map_location=device, weights_only=True)
    return token_bytes
