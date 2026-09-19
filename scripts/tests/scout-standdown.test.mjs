import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

import { standDownReason } from "../scout-standdown.mjs";

test("every stand-down phrasing the prompt has used is accepted", () => {
  const cases = {
    "SCOUT RESULT: nothing filed - every candidate was already covered": "every candidate was already covered",
    "SCOUT-DECISION: none - nothing passed the evidence floor": "nothing passed the evidence floor",
    "SCOUT-DECISION: None - covered by open work": "covered by open work",
    "SCOUT-DECISION: NONE: covered": "covered",
    "scout-decision: none — covered": "covered",
    "SCOUT-DECISION: filed 0": "",
    "SCOUT-DECISION: filed 0 - all covered": "all covered",
    "SCOUT RESULT: Nothing Filed – all covered": "all covered",
    "SCOUT-DECISION: none": "",
    "**SCOUT-DECISION: none - covered**": "covered",
    "`SCOUT RESULT: nothing filed - covered`": "covered",
    "> - SCOUT-DECISION: none - covered": "covered",
  };
  for (const [line, reason] of Object.entries(cases)) {
    assert.equal(standDownReason(`I looked at five candidates.\n\n${line}\n`), reason, line);
  }
});

test("no stand-down line, or a claim of filing, is not a stand-down", () => {
  for (const text of [
    "", "I dispatched four researchers and will wait for them.",
    "SCOUT-DECISION: filed 2", "SCOUT-DECISION: filed 01", "SCOUT-DECISION: nonetheless filed",
    "The line would be SCOUT-DECISION: none - x", "nothing filed",
  ]) {
    assert.equal(standDownReason(text), null, text);
  }
});

test("the last stand-down line wins", () => {
  assert.equal(standDownReason("SCOUT-DECISION: none - first\nSCOUT RESULT: nothing filed - second"), "second");
});

test("the CLI prints the reason and exits 0, or exits 1 with no line", () => {
  const script = fileURLToPath(new URL("../scout-standdown.mjs", import.meta.url));
  const ok = spawnSync(process.execPath, [script], { input: "SCOUT-DECISION: None - covered\n", encoding: "utf8" });
  assert.equal(ok.status, 0);
  assert.equal(ok.stdout.trim(), "covered");
  const bare = spawnSync(process.execPath, [script], { input: "SCOUT-DECISION: filed 0\n", encoding: "utf8" });
  assert.equal(bare.status, 0);
  assert.equal(bare.stdout.trim(), "no reason given");
  const none = spawnSync(process.execPath, [script], { input: "all done\n", encoding: "utf8" });
  assert.equal(none.status, 1);
  assert.equal(none.stdout, "");
});
