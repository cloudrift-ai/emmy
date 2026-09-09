#!/usr/bin/env bash
# Full-attention programs shared by RTX 5090 tracing and measurement.

SEQUENCE_LENGTHS=(1024 2048 4096 8192 16384 32768)

# The sequence lengths one operator and batch is measured at. Every setup is offered except GQA
# prefill at batch 8 and 32768: its query tensor alone is 4.3 GB, the comparison holds it in both
# layouts beside a reference and a candidate output, and that does not fit the 32 GB an RTX 5090
# has. A partial comparison is not evidence, so the setup is withheld rather than measured short.
operator_sequence_lengths() {
  local operator=$1
  local batch=$2
  local length
  for length in "${SEQUENCE_LENGTHS[@]}"; do
    if [ "$operator" = prefill_gqa ] && [ "$batch" -eq 8 ] && [ "$length" -eq 32768 ]; then
      continue
    fi
    printf '%s\n' "$length"
  done
}

operator_code() {
  local operator=$1
  local batch=$2
  local sequence_length=$3
  local q_heads=32
  local kv_heads=32
  local q_length=$sequence_length
  local is_causal=False
  local enable_gqa=False

  case "$operator" in
    prefill_global)
      ;;
    prefill_causal)
      is_causal=True
      ;;
    prefill_gqa)
      q_heads=64
      kv_heads=8
      is_causal=True
      enable_gqa=True
      ;;
    decode_causal)
      q_length=1
      ;;
    decode_gqa)
      q_heads=64
      kv_heads=8
      q_length=1
      ;;
    *)
      echo "unknown operator: $operator" >&2
      return 2
      ;;
  esac

  local attention=
  if [ "$operator" = decode_gqa ]; then
    # Express the query heads per KV head as a broadcast batch dimension. This preserves noncausal
    # decode without materializing repeated K/V tensors.
    attention="F.scaled_dot_product_attention("\
"q.reshape($batch,8,8,1,128),"\
"k.reshape($batch,8,1,$sequence_length,128),"\
"v.reshape($batch,8,1,$sequence_length,128),"\
"is_causal=False).reshape($batch,64,1,128)"
  else
    attention="F.scaled_dot_product_attention("\
"q,k,v,is_causal=$is_causal,enable_gqa=$enable_gqa)"
  fi

  printf '%s' \
    "torch.manual_seed(0);" \
    "q=torch.randn($batch,$q_heads,$q_length,128,dtype=torch.float16);" \
    "k=torch.randn($batch,$kv_heads,$sequence_length,128,dtype=torch.float16);" \
    "v=torch.randn($batch,$kv_heads,$sequence_length,128,dtype=torch.float16);" \
    "$attention"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  if [ "$#" -ne 3 ]; then
    echo "usage: $0 OPERATOR BATCH SEQUENCE_LENGTH" >&2
    exit 2
  fi
  operator_code "$1" "$2" "$3" || exit
  echo
fi
