#!/usr/bin/env bash
# Qwen3-0.6B layer-0 dimensions at c1899de289a04d12100db370d81485cdf75e47ca.
# Complete frontend computations: K/V share one shape; both layer norms share one shape.
set -euo pipefail
operator=$1
seq=$2
norm='norm=lambda x,w: (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)*w;'
rope='rope=lambda x,c,s: x*c+torch.cat((-x[...,64:],x[...,:64]),dim=-1)*s;'
case "$operator" in
  rms_norm)
    echo "${norm}norm(torch.randn(1,$seq,1024,dtype=torch.float16),torch.randn(1024,dtype=torch.float16))" ;;
  q_proj)
    echo "F.linear(torch.randn(1,$seq,1024,dtype=torch.float16),torch.randn(2048,1024,dtype=torch.float16))" ;;
  kv_proj)
    echo "F.linear(torch.randn(1,$seq,1024,dtype=torch.float16),torch.randn(1024,1024,dtype=torch.float16))" ;;
  q_norm_rope|k_norm_rope)
    heads=16
    if [ "$operator" = k_norm_rope ]; then heads=8; fi
    echo "${norm}${rope}(lambda x,w,c,s: rope(norm(x,w).transpose(1,2),c,s))(torch.randn(1,$seq,$heads,128,dtype=torch.float16),torch.randn(128,dtype=torch.float16),torch.randn(1,1,$seq,128,dtype=torch.float16),torch.randn(1,1,$seq,128,dtype=torch.float16))" ;;
  attention)
    echo "F.scaled_dot_product_attention(torch.randn(1,16,$seq,128,dtype=torch.float16),torch.randn(1,8,$seq,128,dtype=torch.float16),torch.randn(1,8,$seq,128,dtype=torch.float16),is_causal=True,enable_gqa=True)" ;;
  o_proj_residual)
    echo "(lambda x,w,r: F.linear(x,w)+r)(torch.randn(1,$seq,2048,dtype=torch.float16),torch.randn(1024,2048,dtype=torch.float16),torch.randn(1,$seq,1024,dtype=torch.float16))" ;;
  gate_up_silu)
    echo "(lambda x,g,u: F.silu(F.linear(x,g))*F.linear(x,u))(torch.randn(1,$seq,1024,dtype=torch.float16),torch.randn(3072,1024,dtype=torch.float16),torch.randn(3072,1024,dtype=torch.float16))" ;;
  down_proj_residual)
    echo "(lambda x,w,r: F.linear(x,w)+r)(torch.randn(1,$seq,3072,dtype=torch.float16),torch.randn(1024,3072,dtype=torch.float16),torch.randn(1,$seq,1024,dtype=torch.float16))" ;;
  *) echo "unknown operator: $operator" >&2; exit 2 ;;
esac
