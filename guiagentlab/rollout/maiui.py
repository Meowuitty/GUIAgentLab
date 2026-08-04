"""Canonical MAI-UI messages and images."""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def normalize_screenshot(image: Image.Image) -> Image.Image:
    """Load a screenshot and normalize the representation shared by train/eval."""
    image.load()
    return image if image.mode == "RGB" else image.convert("RGB")


def decode_screenshot(png: bytes) -> Image.Image:
    """Decode PNG bytes without changing their dimensions."""
    return normalize_screenshot(Image.open(io.BytesIO(png)))


def canonical_prompt_prefix(
    raw_prompt: Sequence[dict[str, Any]] | None,
    instruction: str,
    system_prompt: str,
) -> list[dict[str, Any]]:
    """Return the text prefix used by both OpenAI and VERL transports.

    The text block form matches the historical OpenAI request.  Qwen3-VL's
    processor renders it to the same token sequence as a plain string.
    """
    if raw_prompt:
        system_messages = [item for item in raw_prompt if item.get("role") == "system"]
        if len(system_messages) != 1 or system_messages[0].get("content") != system_prompt:
            raise ValueError("dataset system prompt differs from the pinned MAI-UI prompt")
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [{"type": "text", "text": instruction}],
        },
    ]


def _is_image_message(message: dict[str, Any]) -> bool:
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if isinstance(content, str):
        return "<image>" in content
    return isinstance(content, list) and any(
        isinstance(item, dict) and item.get("type") in {"image", "image_url"} for item in content
    )


def limit_image_history(
    messages: Sequence[dict[str, Any]],
    images: Sequence[Image.Image],
    history_length: int,
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """Keep every assistant action and only the newest screenshots."""
    copied_messages = deepcopy(list(messages))
    copied_images = list(images)
    image_indices = [
        index for index, message in enumerate(copied_messages) if _is_image_message(message)
    ]
    if len(image_indices) != len(copied_images):
        raise ValueError(
            f"chat/image mismatch: {len(image_indices)} messages for {len(copied_images)} images"
        )
    if history_length <= 0 or len(copied_images) <= history_length:
        return copied_messages, copied_images
    remove_count = len(copied_images) - history_length
    remove_indices = set(image_indices[:remove_count])
    return (
        [message for index, message in enumerate(copied_messages) if index not in remove_indices],
        copied_images[remove_count:],
    )


def build_maiui_message_history(
    prefix: Sequence[dict[str, Any]],
    screenshot_count: int,
    assistant_responses: Sequence[str],
    history_length: int,
) -> tuple[list[dict[str, Any]], range]:
    """Build canonical messages without requiring old screenshot pixels.

    MAI-UI retains every previous assistant action while retaining only the
    newest screenshots.  Returning the retained screenshot indices lets
    offline datasets load only images that the model will actually see.
    """
    if screenshot_count != len(assistant_responses) + 1:
        raise ValueError("MAI-UI history requires one more screenshot than assistant response")
    if screenshot_count <= 0:
        raise ValueError("MAI-UI history requires at least one screenshot")
    retained = screenshot_count if history_length <= 0 else min(screenshot_count, history_length)
    first_retained = screenshot_count - retained
    messages = deepcopy(list(prefix))
    for index in range(screenshot_count):
        if index >= first_retained:
            messages.append({"role": "user", "content": [{"type": "image"}]})
        if index < len(assistant_responses):
            messages.append(
                {"role": "assistant", "content": str(assistant_responses[index])}
            )
    return messages, range(first_retained, screenshot_count)


def build_maiui_history(
    prefix: Sequence[dict[str, Any]],
    screenshots: Sequence[Image.Image],
    assistant_responses: Sequence[str],
    history_length: int,
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """Build text + alternating screenshot/action history in one canonical form."""
    messages, retained = build_maiui_message_history(
        prefix,
        len(screenshots),
        assistant_responses,
        history_length,
    )
    images = [normalize_screenshot(screenshots[index]) for index in retained]
    return messages, images


def _image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    normalize_screenshot(image).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def to_openai_messages(
    messages: Sequence[dict[str, Any]], images: Sequence[Image.Image]
) -> list[dict[str, Any]]:
    """Adapt canonical image placeholders to OpenAI image URLs losslessly."""
    output = deepcopy(list(messages))
    image_index = 0
    for message in output:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item_index, item in enumerate(content):
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            if image_index >= len(images):
                raise ValueError("not enough images for OpenAI message placeholders")
            content[item_index] = {
                "type": "image_url",
                "image_url": {"url": _image_data_url(images[image_index])},
            }
            image_index += 1
    if image_index != len(images):
        raise ValueError("unused images remain after OpenAI message conversion")
    return output


def gui_state_fingerprint(image: Image.Image) -> int:
    """Return a status-bar-resistant 64-bit perceptual GUI fingerprint.

    A difference hash is stable across PNG re-encoding and tiny pixel noise.
    Cropping the Android status/navigation bars prevents clock/network changes
    from splitting otherwise identical GUI states.
    """
    normalized = normalize_screenshot(image)
    width, height = normalized.size
    top = min(height - 1, round(height * 0.05))
    bottom = max(top + 1, round(height * 0.97))
    content = normalized.crop((0, top, width, bottom)).convert("L")
    reduced = content.resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(reduced, dtype=np.int16)
    bits = (pixels[:, 1:] >= pixels[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    # Torch int64 is signed; retain all 64 bits through two's-complement form.
    return value if value < (1 << 63) else value - (1 << 64)


def gui_state_detail_fingerprint(image: Image.Image) -> int:
    """Return a 64-bit spatial-detail fingerprint for anchor refinement.

    An edge map is split into four tiles and each tile contributes a 16-bit
    difference hash. Localizing the edge comparisons preserves small text and
    control changes that disappear in the whole-screen 64-bit fingerprint.
    """
    normalized = normalize_screenshot(image)
    width, height = normalized.size
    top = min(height - 1, round(height * 0.05))
    bottom = max(top + 1, round(height * 0.97))
    content = normalized.crop((0, top, width, bottom)).convert("L").filter(ImageFilter.FIND_EDGES)
    content_width, content_height = content.size
    value = 0
    for row in range(2):
        for column in range(2):
            tile = content.crop(
                (
                    column * content_width // 2,
                    row * content_height // 2,
                    (column + 1) * content_width // 2,
                    (row + 1) * content_height // 2,
                )
            )
            reduced = tile.resize((5, 4), Image.Resampling.LANCZOS)
            pixels = np.asarray(reduced, dtype=np.int16)
            bits = (pixels[:, 1:] >= pixels[:, :-1]).reshape(-1)
            for bit in bits:
                value = (value << 1) | int(bit)
    return value if value < (1 << 63) else value - (1 << 64)


def fingerprint_hamming_distance(left: int, right: int) -> int:
    """Compute Hamming distance for signed 64-bit fingerprints."""
    mask = (1 << 64) - 1
    return (((int(left) & mask) ^ (int(right) & mask)) & mask).bit_count()
