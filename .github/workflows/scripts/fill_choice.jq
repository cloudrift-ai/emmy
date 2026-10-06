# The hardware golden to fill: among the files `emmy golden list --missing` names whose card is free ($available, from
# `emmy vm available`), the one with the most missing measurements; `{}` when no card is free.
[group_by(.file)[] | select(.[0].gpu as $gpu | $available[0] | index($gpu))
  | {file: .[0].file, gpu: .[0].gpu, rows: length}]
| sort_by(-.rows, .file) | .[0] // {}
