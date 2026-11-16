# One line on an `emmy golden realize --json` report over the measurement freeze: the rows that compile under their
# own schedule with the card's file as the evidence, and per card the rows the compiler no longer realizes.
([.[].rows] | add // 0) as $rows
| ([.[].failed | length] | add // 0) as $failed
| ([.[] | select((.failed | length) > 0) | "\(.gpu_name) \(.failed | length) of \(.rows)"] | join(", ")) as $cards
| "Measurement freeze: \($rows - $failed) of \($rows) rows realize"
  + (if $failed > 0 then ". Not realized: \($cards)" else "" end)
