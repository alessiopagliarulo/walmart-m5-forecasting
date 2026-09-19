#!/usr/bin/env node
/**
 * Did the Scout deliberately file nothing? Reads the agent's own final words on stdin
 * and finds its stand-down line.
 *
 * The Scout's verify step fails a run that filed zero proposals and never said why:
 * that is an agent that backgrounded its researchers and ended its turn. A run that
 * filed nothing ON PURPOSE must stay green, so this accepts every phrasing the Scout's
 * prompt has told it to use, however the model dresses it up:
 *
 *   SCOUT RESULT: nothing filed - <reason>
 *   SCOUT-DECISION: none - <reason>
 *   SCOUT-DECISION: filed 0
 *
 * Case never matters ("None", "NONE", "Nothing Filed"), the separator before the
 * reason may be "-", ":", a long dash or nothing, the reason itself is optional, and
 * the line may be wrapped in markdown (`code`, **bold**, a > quote or a - bullet).
 * The last such line wins. A line claiming it filed something ("filed 2") is not a
 * stand-down: if the count says zero, that is still a malfunction.
 *
 *   node scripts/scout-standdown.mjs < final-message.txt
 *       Prints the reason (or "no reason given") and exits 0 on a stand-down line;
 *       prints nothing and exits 1 when there is none.
 */
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const STAND_DOWN =
  /^SCOUT[\s_-]*(?:RESULT|DECISION)\s*:\s*(?:nothing\s+filed|filed\s+(?:0|zero|nothing)|none)(?![\w])\s*(?:[-:–—.,;]+\s*)?(.*)$/i;

/** The stand-down reason from the Scout's final text, "" when it gave none, or null. */
export function standDownReason(text) {
  let found = null;
  for (const raw of String(text ?? "").split(/\r?\n/)) {
    const line = raw
      .trim()
      .replace(/^(?:>\s*)*(?:[-*+]\s+)?/, "")
      .replace(/^[`*_\s]+/, "")
      .replace(/[`*_\s]+$/, "");
    const m = STAND_DOWN.exec(line);
    if (m) found = m[1].replace(/[`*_\s]+$/, "").trim();
  }
  return found;
}

const invokedDirectly = (() => {
  try {
    return import.meta.url === pathToFileURL(process.argv[1] ?? "").href;
  } catch {
    return false;
  }
})();

if (invokedDirectly) {
  const reason = standDownReason(readFileSync(0, "utf8"));
  if (reason === null) process.exit(1);
  console.log(reason || "no reason given");
}
