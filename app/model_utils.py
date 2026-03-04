# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, ast, operator, torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
except ModuleNotFoundError:
    _BNB_AVAILABLE = False

MODEL_ID   = os.getenv("MODEL_ID", "microsoft/phi-1_5")
SYS_PROMPT = (
    "You are a meticulous mathematician. "
    "When asked to compute or simplify an arithmetic expression, ALWAYS list "
    "each step on a new line: 'Step 1:', 'Step 2:', … and finish with "
    "'Final Answer:'. For other questions, give a concise, accurate answer."
)

MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.2
TOP_P          = 0.95

def load_model():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    if _BNB_AVAILABLE and device != "cpu":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto" if device != "cpu" else None,
            trust_remote_code=True,
        )

    model.eval()
    return model, tok, device

_ALLOWED_OPS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,
    ast.USub: operator.neg,
}

def _eval_node(node):
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp)   and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](
            _eval_node(node.left), _eval_node(node.right)
        )
    raise ValueError("disallowed expression")

def safe_arith(expr: str) -> float | None:
    if not re.fullmatch(r"[0-9\.\s\+\-\*\/\(\)\^]+", expr):
        return None
    expr = expr.replace("^", "**")
    try:
        tree = ast.parse(expr, mode="eval")
        return _eval_node(tree.body)
    except Exception:
        return None

def maybe_solve_direct(user_msg: str) -> str | None:
    ans = safe_arith(user_msg.strip())
    if ans is None:
        return None
    final = int(ans) if ans == int(ans) else ans
    return f"Step 1-n: (computed directly)\nFinal Answer: {final}"

def generate_response(model, tok, user_msg: str) -> str:
    direct = maybe_solve_direct(user_msg)
    if direct is not None:
        return direct

    prompt = (
        f"<s>[INST] <<SYS>>\n{SYS_PROMPT}\n<</SYS>>\n\n"
        f"{user_msg.strip()}\n[/INST]"
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
            eos_token_id=tok.eos_token_id,
        )

    reply = tok.decode(
        output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True
    ).strip()
    return reply
