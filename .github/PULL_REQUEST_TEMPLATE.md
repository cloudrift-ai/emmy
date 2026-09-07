<!--
Title: a functional description, readable with no context. "Fix X", "Optimize Y", "Do X because Y".
Not a component name, not a branch name, not a ticket id.

Write this body, then revise it at least twice before posting. Each pass: read it as a reviewer who has no context,
check it against the rules below and against the design philosophy in AGENTS.md, and cut. A first draft is always too
long. Stop when nothing else can come out without losing the point.

Do not hard-wrap the text you write here. GitHub wraps it for the reader, and manual line breaks only make the
body hard to edit. The ~120-character rule applies to files in the repository, not to a pull-request body.
-->

## Abstract

The checked-in bucketing baseline gained four entries spelled without the marker suffix a CUDA test is collected under. A key that never matches is the same as no key at all, so the staleness gate goes on naming those four tests as missing and fails the run it just balanced around them. This spells them the way the gate reads them, and records the one serving-runner pole that crossed five seconds on its own.

---

## Why the ids differed

A subset run and the full suite disagree on what a node id is. `make test` runs under `-n auto --dist=loadgroup`, where the xdist group marker lands in the id (`…::test_x@cuda`); a bare `pytest path::test_x` reports no suffix. The four entries were recorded from subset runs, which is how they went in without one — and why the gate kept asking for tests the file appeared to list.

The baseline already held 152 keys carrying the suffix, so the file's own convention was never in doubt.

## Evidence

Under `-n 2 --dist=loadgroup` — the shape `make test` uses — the gate is quiet on the affected file. Under a bare subset run it still complains, as it did before this change: a subset sees subset ids, and this file is a baseline for the full suite.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01NHvyWiofb3RYvVoDzqSeoQ
