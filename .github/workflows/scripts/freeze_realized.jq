# One line on an `emmy golden realize --json` report over the measurement freeze: the rows that compile under their
# own schedule with the card's file as the evidence, and per card the rows that do not — among them the rows whose
# kernel no repository golden of the card reaches, which are never compiled.
([.[].rows] | add // 0) as $rows
| ([.[].failed | length] | add // 0) as $failed
| ([.[] | select((.failed | length) > 0)
    | (.failed | map(select(.reason | startswith("no repository golden"))) | length) as $unreached
    | "\(.gpu_name) \(.failed | length) of \(.rows)" + (if $unreached > 0 then " (\($unreached) reached by no golden)" else "" end)]
   | join(", ")) as $cards
| "Measurement freeze: \($rows - $failed) of \($rows) rows realize"
  + (if $failed > 0 then ". Not realized: \($cards)" else "" end)
