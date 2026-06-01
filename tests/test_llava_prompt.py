from __future__ import annotations

from paper_reproduce.models.llava_hf import _format_llava_prompt


class FakeTemplateProcessor:
    def __init__(self) -> None:
        self.messages = None

    def apply_chat_template(self, messages, *, add_generation_prompt: bool, tokenize: bool) -> str:
        self.messages = messages
        assert add_generation_prompt is True
        assert tokenize is False
        return "templated prompt"


class BrokenTemplateProcessor:
    def apply_chat_template(self, messages, *, add_generation_prompt: bool, tokenize: bool) -> str:
        raise ValueError("unsupported template")


def test_llava_prompt_prefers_processor_chat_template() -> None:
    processor = FakeTemplateProcessor()

    prompt = _format_llava_prompt(processor, "Describe the image.")

    assert prompt == "templated prompt"
    assert processor.messages == [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the image."},
            ],
        }
    ]


def test_llava_prompt_preserves_preformatted_chat_prompt() -> None:
    prompt = "USER: <image>\nDescribe the image.\nASSISTANT:"

    assert _format_llava_prompt(FakeTemplateProcessor(), prompt) == prompt


def test_llava_prompt_falls_back_to_legacy_template() -> None:
    assert (
        _format_llava_prompt(BrokenTemplateProcessor(), "Describe the image.")
        == "USER: <image>\nDescribe the image.\nASSISTANT:"
    )
