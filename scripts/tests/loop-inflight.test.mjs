import assert from "node:assert/strict";
import { test } from "node:test";

import { explicitCovers, referencedIssues } from "../loop-inflight.mjs";

const REPO = "alessiopagliarulo/walmart-m5-forecasting";
const refs = (...texts) => referencedIssues(REPO, ...texts).sort((a, b) => a - b);

test("every GitHub closing keyword form counts", () => {
  for (const kw of ["close", "closes", "closed", "fix", "fixes", "fixed", "resolve", "resolves", "resolved"]) {
    assert.deepEqual(refs(`${kw} #12`), [12], kw);
    assert.deepEqual(refs(`${kw.toUpperCase()}: #12`), [12], `${kw} upper with colon`);
  }
  assert.deepEqual(refs("Fixes #3, closes #4 and resolves #5."), [3, 4, 5]);
  assert.deepEqual(refs("closes#7"), [7]);
});

test("closing references to this repository count, by owner/name or issue URL", () => {
  assert.deepEqual(refs(`closes ${REPO}#12`), [12]);
  assert.deepEqual(refs("closes AlessioPagliarulo/Walmart-M5-Forecasting#13"), [13]);
  assert.deepEqual(refs(`fixes https://github.com/${REPO}/issues/14`), [14]);
});

test("passing mentions never count", () => {
  for (const text of [
    "see #12", "for #12", "issue #12", "idea #12", "towards #12", "toward #12", "part of #12",
    "refs #12", "ref #12", "references #12", "related to #12", "step #2", "#12",
    `https://github.com/${REPO}/issues/12`, `see https://github.com/${REPO}/issues/12`,
    "fm/issue-12-thing", "claude/issue-12", "prefixes #12", "unfixed #12", "closes #12abc",
    `closes https://github.com/${REPO}/pull/12`,
  ]) {
    assert.deepEqual(refs(text), [], text);
  }
});

test("issues in other repositories never count, even with a closing keyword", () => {
  assert.deepEqual(refs("closes someone/else#12"), []);
  assert.deepEqual(refs("fixes https://github.com/someone/else/issues/12"), []);
  assert.deepEqual(refs(`fixes https://github.com/${REPO}-fork/issues/12`), []);
  assert.deepEqual(refs("see https://github.com/someone/else/issues/12"), []);
});

test("without a repository only the short form counts", () => {
  assert.deepEqual(referencedIssues(null, "closes #1", `closes ${REPO}#2`), [1]);
});

test("a PR that only mentions an idea does not cover it", () => {
  const data = {
    openPrs: [
      { kind: "pr", url: "u1", loop: false, refs: referencedIssues(REPO, "Tweak plots", "Part of #12, see #12") },
      { kind: "pr", url: "u2", loop: false, refs: referencedIssues(REPO, "Hurdle model", "Closes #13") },
    ],
    branches: [], mergedPrs: [], commits: [],
  };
  assert.deepEqual(explicitCovers({ number: 12 }, data), []);
  assert.deepEqual(explicitCovers({ number: 13 }, data).map((c) => c.item.url), ["u2"]);
});
