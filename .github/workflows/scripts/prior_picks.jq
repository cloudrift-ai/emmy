# One line on an `emmy eval prior --json` report's pools: how many re-decided pools reproduce their golden, and the
# regret of the picks a golden row measured.
[.pools // [] | .[] | select(.error == null)] as $pools
| ($pools | map(select(.matched == .total)) | length) as $exact
| ([$pools[] | select(.regret != null)] | sort_by(.regret)) as $measured
| "Prior picks: \($exact) of \($pools | length) pools reproduce their golden exactly; \($measured | length) picks measured"
  + if ($measured | length) > 0 then
      ", median regret \($measured[($measured | length) / 2 | floor].regret)x, worst \($measured[-1].regret)x (\($measured[-1].pool))"
    else "" end
