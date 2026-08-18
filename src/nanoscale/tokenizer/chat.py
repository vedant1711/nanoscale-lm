"""Chat templating and completion masking for SFT (spec B1 / B5).

The template is deliberately minimal and explicit:

.. code-block:: text

    <bos><system>SYSTEM TEXT<eot><user>USER TEXT<eot><assistant>REPLY<eot><eos>

Role tokens are *special* tokens: they are only ever produced when the caller
explicitly renders a conversation, never by encoding user text (see
:meth:`~nanoscale.tokenizer.bpe.BPETokenizer.encode`). A user cannot type the literal
string ``<assistant>`` and have it become a turn boundary.

Alongside the token IDs, :func:`render_chat` returns a **completion mask**: 1 on the
tokens the model is trained to produce, 0 on prompt tokens. Spec B5 requires the SFT
loss to be masked to completion tokens only, and Phase 6 tests that masking directly.
The mask covers the assistant's reply *and* its terminating ``<eot>`` (the model must
learn to stop) but not the ``<assistant>`` role token that opens the turn (that is a
prompt cue supplied by the harness).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from nanoscale.tokenizer.bpe import BPETokenizer

__all__ = ["ChatExample", "Message", "Role", "render_chat", "render_prompt"]

Role = Literal["system", "user", "assistant"]

_ROLE_TOKEN: dict[str, str] = {
    "system": "<system>",
    "user": "<user>",
    "assistant": "<assistant>",
}


@dataclass(frozen=True, slots=True)
class Message:
    """One turn of a conversation."""

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class ChatExample:
    """A rendered conversation: token IDs plus the SFT completion mask."""

    ids: list[int]
    completion_mask: list[int]

    def __post_init__(self) -> None:
        """Validate that the mask lines up with the token stream."""
        if len(self.ids) != len(self.completion_mask):
            raise ValueError(
                f"ids ({len(self.ids)}) and completion_mask ({len(self.completion_mask)}) "
                "must have the same length."
            )

    @property
    def n_completion_tokens(self) -> int:
        """Number of supervised (loss-carrying) tokens."""
        return sum(self.completion_mask)

    def __len__(self) -> int:
        """Total token count."""
        return len(self.ids)


def render_chat(
    tokenizer: BPETokenizer,
    messages: Sequence[Message],
    *,
    add_bos: bool = True,
    add_eos: bool = True,
    train_on_all_assistant_turns: bool = True,
) -> ChatExample:
    """Render a conversation into token IDs and an SFT completion mask.

    Args:
        tokenizer: A trained tokenizer (its special tokens define the role markers).
        messages: The conversation, in order.
        add_bos: Prepend ``<bos>``.
        add_eos: Append ``<eos>`` after the final turn.
        train_on_all_assistant_turns: If True, every assistant turn is supervised. If
            False, only the last one is, which is what you want when a multi-turn
            example was constructed by appending to a fixed transcript.

    Returns:
        A :class:`ChatExample`.

    Raises:
        ValueError: If ``messages`` is empty or a role is unknown.
    """
    if not messages:
        raise ValueError("Cannot render an empty conversation.")

    last_assistant = max(
        (i for i, m in enumerate(messages) if m.role == "assistant"),
        default=-1,
    )

    ids: list[int] = []
    mask: list[int] = []

    def emit(chunk: Sequence[int], supervised: bool) -> None:
        ids.extend(chunk)
        mask.extend([1 if supervised else 0] * len(chunk))

    if add_bos:
        emit([tokenizer.bos_id], False)

    for index, message in enumerate(messages):
        role_token = _ROLE_TOKEN.get(message.role)
        if role_token is None:
            raise ValueError(f"Unknown role {message.role!r}; expected {sorted(_ROLE_TOKEN)}.")
        # The role marker is always a prompt cue, never a training target.
        emit([tokenizer.special_to_id[role_token]], False)

        supervised = message.role == "assistant" and (
            train_on_all_assistant_turns or index == last_assistant
        )
        body = tokenizer.encode(message.content, allowed_special=False)
        emit(body, supervised)
        # The model must learn where to stop, so <eot> is supervised with the reply.
        emit([tokenizer.eot_id], supervised)

    if add_eos:
        emit([tokenizer.eos_id], False)

    return ChatExample(ids=ids, completion_mask=mask)


def render_prompt(
    tokenizer: BPETokenizer,
    messages: Sequence[Message],
    *,
    add_bos: bool = True,
) -> list[int]:
    """Render a conversation as a *generation prompt*.

    Identical to :func:`render_chat` except that it ends with an open
    ``<assistant>`` turn; the model is expected to continue from there. This is the
    exact prefix used at inference time, which keeps train/serve formatting in sync.
    """
    if not messages:
        raise ValueError("Cannot render an empty conversation.")
    ids: list[int] = []
    if add_bos:
        ids.append(tokenizer.bos_id)
    for message in messages:
        role_token = _ROLE_TOKEN.get(message.role)
        if role_token is None:
            raise ValueError(f"Unknown role {message.role!r}; expected {sorted(_ROLE_TOKEN)}.")
        ids.append(tokenizer.special_to_id[role_token])
        ids.extend(tokenizer.encode(message.content, allowed_special=False))
        ids.append(tokenizer.eot_id)
    ids.append(tokenizer.special_to_id["<assistant>"])
    return ids
