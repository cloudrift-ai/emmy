# One line on an unfinished discover-models run, from its agent events: how many tool calls finished, how many recipe
# notes it wrote, and the start of its last text.
([.[] | select(.part.tool != null and .part.state.status == "completed")]) as $tools
| ([$tools[] | .part.state.input.filePath // .part.state.input.path // empty | select(test("DISCOVERY\\.md$"))] | unique) as $notes
| ([.[] | select(.type == "text") | .part.text][-1] // "") as $last
| "Unfinished after \($tools | length) tool calls; \($notes | length) discovery note(s) written"
  + if ($last | length) > 0 then "; last text: \($last | gsub("\\s+"; " ") | .[0:200])" else "" end
