# One line on `emmy golden list --json` over the repository goldens and the realization corpus, and the corpus's
# expected-failure cases: how many rows timed beside torch.compile are slower, the slowest three, the known gaps.
def behind: [.[] | select(.vs_tcompile != null)] as $timed
  | ($timed | map(select(.vs_tcompile > 1)) | sort_by(-.vs_tcompile)) as $slow
  | "\($slow | length) of \($timed | length) rows timed beside torch.compile are slower"
    + if ($slow | length) > 0 then
        " (" + ([$slow[:3][] | "\(.row) \(.vs_tcompile * 100 | round / 100)x"] | join(", ")) + ")"
      else "" end;
"Goldens: \($goldens[0] | behind). Corpus: \($corpus[0] | behind). Realization gaps: \($xfail[0] | length) expected-failure case(s)"
+ if ($xfail[0] | length) > 0 then " (" + ($xfail[0] | map(split("/")[-1]) | join(", ")) + ")" else "" end
