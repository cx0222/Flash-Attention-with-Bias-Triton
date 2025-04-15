"""
Attention functions for triton, torch and sdpa

Refactor from the original project from [https://github.com/pengzhangzhi]

@author: Zhangzhi Peng, Xuan Chen
"""

from flash_attn_w_bias import FlashAttnFunc

attention_triton = FlashAttnFunc.apply


def attention_torch(q, k, v, softmax_scale, bias=None):
    """
    Implements standard scaled dot-product attention using PyTorch.

    Args:
        q: Query tensor of shape (batch_size, seqlen_q, nheads, headdim)
        k: Key tensor of shape (batch_size, seqlen_k, nheads, headdim)
        v: Value tensor of shape (batch_size, seqlen_k, nheads, headdim)
        softmax_scale: Scalar for softmax normalization (default: 1/sqrt(headdim))
        bias: Optional bias tensor, shape (batch, nheads, seqlen_q, seqlen_k)

    Returns:
        Output tensor of shape (batch_size, seqlen_q, nheads, headdim)
    """
    # Compute scaled dot-product attention scores
    attn_scores = torch.einsum("bqnh,bknh->bqnk", q, k)  # (batch, seqlen_q, nheads, seqlen_k)
    # Apply softmax scaling
    attn_scores *= softmax_scale

    # Apply bias if provided
    if bias is not None:
        attn_scores += bias.permute(0, 2, 1, 3)

    # Apply softmax along key dimension
    attn_probs = torch.softmax(attn_scores, dim=-1)

    # Compute output by weighted sum of values
    output = torch.einsum("bqnk,bknh->bqnh", attn_probs, v)

    return output


def attention_sdpa(q, k, v, softmax_scale, bias=None):
    """
    Implements scaled dot-product attention using PyTorch's F.scaled_dot_product_attention.

    Args:
        q: Query tensor of shape (batch_size, seqlen_q, nheads, headdim)
        k: Key tensor of shape (batch_size, seqlen_k, nheads, headdim)
        v: Value tensor of shape (batch_size, seqlen_k, nheads, headdim)
        softmax_scale: Scalar for softmax normalization (default: 1/sqrt(headdim))
        bias: Optional bias tensor, shape (batch, nheads, seqlen_q, seqlen_k)

    Returns:
        Output tensor of shape (batch_size, seqlen_q, nheads, headdim)
    """
    # SDPA expects input of shape (B, Nh, L, D)
    # Our inputs are already in this format after transposing
    q = q.transpose(1, 2)  # (batch, nheads, seqlen_q, headdim)
    k = k.transpose(1, 2)  # (batch, nheads, seqlen_k, headdim)
    v = v.transpose(1, 2)  # (batch, nheads, seqlen_k, headdim)

    # Prepare attention mask from bias
    attn_mask = bias if bias is not None else None

    # Call SDPA with our inputs
    output = torch.nn.functional.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=softmax_scale,
        is_causal=False
    )

    # Return to original format (B, Lq, Nh, D)
    return output.transpose(1, 2)
