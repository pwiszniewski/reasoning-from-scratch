import os
from pathlib import Path

import torch
import chainlit

from reasoning_from_scratch.ch02 import (
    get_device,
)
from reasoning_from_scratch.ch02 import generate_text_basic_stream_cache
from reasoning_from_scratch.ch03 import (
    load_model_and_tokenizer,
    load_tokenizer_only
)
from reasoning_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B



WHICH_MODEL = "reasoning"
MAX_NEW_TOKENS = 38912
LOCAL_DIR = "qwen3"
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")
COMPILE = False


DEVICE = get_device()


def load_app_model_and_tokenizer():
    if CHECKPOINT_PATH is None:
        return load_model_and_tokenizer(
            which_model=WHICH_MODEL,
            device=DEVICE,
            use_compile=COMPILE,
            local_dir=LOCAL_DIR,
        )

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}"
        )

    tokenizer = load_tokenizer_only(
        which_model=WHICH_MODEL, local_dir=LOCAL_DIR
    )
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu")
    )
    model.to(DEVICE)

    if COMPILE:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)

    return model, tokenizer


MODEL, TOKENIZER = load_app_model_and_tokenizer()

EOS_TOKEN_IDS = (
    TOKENIZER.encode("<|im_end|>")[0],
    TOKENIZER.encode("<|endoftext|>")[0]
)


@chainlit.on_chat_start
async def on_start():
    chainlit.user_session.set("history", [])
    chainlit.user_session.get("history").append(
        {"role": "system", "content": "You are a helpful assistant."}
    )


@chainlit.on_message
async def main(message: chainlit.Message):

    input_ids = TOKENIZER.encode(message.content)
    input_ids_tensor = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)


    out_msg = chainlit.Message(content="")
    await out_msg.send()


    for tok in generate_text_basic_stream_cache(
        model=MODEL,
        token_ids=input_ids_tensor,
        max_new_tokens=MAX_NEW_TOKENS,
    ):
        token_id = tok.squeeze(0).item()
        if token_id in EOS_TOKEN_IDS:
            break
        piece = TOKENIZER.decode([token_id])
        await out_msg.stream_token(piece)


    await out_msg.update()

    # chainlit run app.py