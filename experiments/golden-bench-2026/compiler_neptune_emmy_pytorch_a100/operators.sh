#!/usr/bin/env bash
# The common attention operators: the sequence sweep and the inline module source for each one.
# run_emmy.sh sources this file for the sweep, and the trace that produces a committed golden reads
# its module source from here (`./operators.sh OPERATOR SEQUENCE_LENGTH`), so the tuned program and
# the benched program are the same program by construction.

SEQUENCE_LENGTHS=(256 512 1024 2048 4096 8192 16384 32768)

operator_code() {
  local operator=$1
  local sequence_length=$2
  local q_heads=32
  local kv_heads=32
  local q_length=$sequence_length
  local is_causal=False
  local enable_gqa=False
  local q=
  local k=
  local v=

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

  q="torch.randn(1,$q_heads,$q_length,128,dtype=torch.float16)"
  k="torch.randn(1,$kv_heads,$sequence_length,128,dtype=torch.float16)"
  v="torch.randn(1,$kv_heads,$sequence_length,128,dtype=torch.float16)"

  local q_input=$q
  local k_input=$k
  local v_input=$v
  case "$operator" in
    decode_*)
      # Named operands preserve decode's complete trace, whether its selected placement is fused or split.
      q_input=q
      k_input=k
      v_input=v
      ;;
  esac

  local attention=
  if [ "$operator" = decode_gqa ]; then
    # Express the eight query heads per KV head as a broadcast batch dimension. This preserves
    # noncausal decode semantics: torch.export drops is_causal=False but retains enable_gqa=True,
    # which older Emmy frontends can otherwise mistake for a causal flag.
    attention="F.scaled_dot_product_attention("\
"$q_input.reshape(1,8,8,1,128),"\
"$k_input.reshape(1,8,1,$sequence_length,128),"\
"$v_input.reshape(1,8,1,$sequence_length,128),"\
"is_causal=False).reshape(1,64,1,128)"
  else
    attention="F.scaled_dot_product_attention("\
"$q_input,$k_input,$v_input,is_causal=$is_causal,enable_gqa=$enable_gqa)"
  fi

  case "$operator" in
    prefill_*)
      # Prefill recognizes the value channel as part of the online-softmax carrier only while the
      # three SDPA operands remain direct expression inputs.
      printf '%s' "torch.manual_seed(0);$attention"
      ;;
    decode_*)
      printf '%s' \
        "torch.manual_seed(0);" \
        "q=$q;" \
        "k=$k;" \
        "v=$v;" \
        "$attention"
      ;;
  esac
}

# Executed rather than sourced: print one setup's module source, ready for `emmy trace --code`.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  if [ "$#" -ne 2 ]; then
    echo "usage: $0 OPERATOR SEQUENCE_LENGTH" >&2
    exit 2
  fi
  operator_code "$1" "$2" || exit
  echo
fi
