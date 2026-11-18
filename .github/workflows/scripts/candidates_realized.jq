# One phrase on an `emmy candidates realize --json` report: the candidate rows that compile under their own schedule.
([.[].rows] | add // 0) as $rows
| ([.[].failed | length] | add // 0) as $failed
| "\($rows - $failed) of \($rows) rows realize"
