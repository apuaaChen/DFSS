import torch
import torch.nn.functional as F
import pydfss

# problem sizes
num_heads = 4
head_dim = 64
sequence_length = [256, 512, 1024, 2048, 4096]


def full_attention(query, key, value):
    qk = torch.bmm(query, key)
    softmax = F.softmax(qk, dim=-1)
    av = torch.bmm(softmax, value)
    return av

def dfss_attention(query, key, value):
    qk_nnz, qk_meta = pydfss.sddmm(
        query, key, None, False
    )

    softmax = F.softmax(qk_nnz, dim=-1)
    av = pydfss.spmm(softmax, qk_meta, value, False)
    return av

speedups =[]

for length in sequence_length:
    if length < 4096: batch_size = 48
    else: batch_size  = 24

    query = torch.randn(size=(batch_size * num_heads, length, head_dim), dtype=torch.bfloat16, device="cuda")
    key = torch.randn(size=(batch_size * num_heads, length, head_dim), dtype=torch.bfloat16, device="cuda")
    key_t = torch.randn(size=(batch_size * num_heads, head_dim, length), dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(query)


    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # warmup
    for i in range(100):
        full_attention(query, key_t, value)

    start.record()
    for i in range(100):
        full_attention(query, key_t, value)
    end.record()
    torch.cuda.synchronize()
    full_attention_time = start.elapsed_time(end)

    # warmup
    for i in range(100):
        dfss_attention(query, key, value)

    start.record()
    for i in range(100):
        dfss_attention(query, key, value)
    end.record()
    torch.cuda.synchronize()
    dfss_time = start.elapsed_time(end)

    speedups.append(full_attention_time / dfss_time)

print("attention speedup: %.2f ~ %.2f" % (min(speedups), max(speedups)))

