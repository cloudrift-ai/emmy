# One line on an `emmy eval prior --pools measured --json` report: per card, what the prior's pick costs against the
# measured best — regret@1 (the pick ships) and, where a pool holds more than ten rows, regret@10 (bench the top ten,
# keep the best) — median and worst over the pools each covers (n), then the rows the admission rule dropped.
def x: (. * 100 | round) / 100;
def regret(name; block):
  if block.median == null then empty
  else "\(name) \(block.median | x)x median, \(block.worst | x)x worst (\(block.groups))" end;
([.summaries[] | select(.metrics.regret1.median != null)
  | "\(.axes.gpu): " + ([regret("regret@1"; .metrics.regret1), regret("regret@10"; .metrics.regret10)] | join("; "))]
 | join(". ")) as $cards
| "Measured pools: \(.header.groups) over \(.header.rows) rows"
  + (if ($cards | length) > 0 then ". " + $cards else "; no pool holds two rows" end)
  + (if ((.header.dropped // {}) | length) > 0
     then ". Dropped before grouping: " + ([.header.dropped | to_entries[] | "\(.value) \(.key)"] | join(", "))
     else "" end)
