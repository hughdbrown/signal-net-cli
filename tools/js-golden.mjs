// Runs the browser lab's own <script> headless (DOM stubbed) and prints golden values as JSON.
// Usage: node tools/js-golden.mjs "<path to Signal_Network_Lab ... .html>" [sigTicks] [twoEpochs] [onlyScenario]
// Regenerate the fixture: node tools/js-golden.mjs "../Signal_Network_Lab_3_11_4 9.TWO-STAGE.html" > tests/fixtures/golden.json
import { readFileSync } from "node:fs";

const [, , htmlPath, sigTicksArg = "3000", twoEpochsArg = "3"] = process.argv;
const html = readFileSync(htmlPath, "utf8");
const script = html.slice(html.lastIndexOf("<script>") + 8, html.lastIndexOf("</script>"));

const defaults = {
  tmode: "meaning", hc: "300000", no: "0", ip: "50", tp: "20", src: "list", focus: "cur", eng: "sig",
  clev: "30", spr: "30", spw: "12", spt: "40000", clrole: "pairs",
};
const checkedOn = new Set(["adapt", "unl", "seedw", "spon", "smart"]);
const els = new Map();
function el(id) {
  const ctx = {
    createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
    putImageData() {}, fillRect() {}, fillStyle: "",
  };
  const e = {
    id, value: defaults[id] ?? "", checked: checkedOn.has(id), textContent: "", innerHTML: "",
    children: [], style: {}, dataset: {}, options: [],
    prepend(c) { this.children.unshift(c); },
    get lastChild() { const self = this; return { remove() { self.children.pop(); } }; },
    appendChild(c) { this.children.push(c); },
    getContext: () => ctx, querySelectorAll: () => [], addEventListener() {}, click() {},
  };
  if (id === "tmode") e.options = ["cur2", "resp", "meaning", "lang"].map((value) => ({ value }));
  return e;
}
globalThis.document = {
  getElementById: (id) => (els.has(id) || els.set(id, el(id)), els.get(id)),
  createElement: () => el("_"),
};
globalThis.window = globalThis;
globalThis.requestAnimationFrame = () => {};
globalThis.setInterval = () => {};
globalThis.MessageChannel = class { constructor() { this.port1 = {}; this.port2 = { postMessage() {} }; } };
globalThis.localStorage = { getItem: () => null, setItem() {} };

// Scenario code runs inside the page's script scope after start-up; each scenario gets a fresh page.
const common = `
  const sumW = () => { let s = 0; for (let j = 0; j < w.length; j++) s += w[j]; return s; };
  const sumTag = () => { let s = 0; for (let j = 0; j < tag.length; j++) s = (s + tag[j]) % 4294967296; return s; };
  const sumTgt = () => { let s = 0; for (let j = 0; j < tgt.length; j++) s = (s + tgt[j]) % 4294967296; return s; };
  const logs = () => $("tlog").children.map((c) => c.textContent).reverse();
  const run = (n) => { for (let t = 0; t < n; t++) { TH = 0.5 + 2.5 * S(); step(); } };
  const state = () => ({ tick, ca, seed: seed >>> 0, em, epoch, cur, streak, pt, spawnTot, drT, capHits,
    smFix: SM.fix, smPen: SM.pen, sumW: sumW(), sumTag: sumTag(), sumTgt: sumTgt(),
    accSum: accS.reduce((a, b) => a + b, 0), log: logs() });
  const fnv = (s) => { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; } return h; };
`;
const scenarios = {
  basic: `
    const out = {};
    out.startup = { seed: seed >>> 0, npairs: pairs.length, pairs: pairs.map((p) => p[0] + " => " + p[1]),
      order: order.slice(), sumW: sumW(), sumTag: sumTag(), sumTgt: sumTgt() };
    out.hash = [hash(0, 0), hash(1, 2), hash(4000000000, 77), hs("hello world"), pay(104, 3)];
    $("tlog").children.length = 0;
    $("train").onclick({ target: $("train") });
    run(${+sigTicksArg});
    out.sig = { ticks: ${+sigTicksArg}, ...state() };
    $("tlog").children.length = 0;
    $("train").onclick({ target: $("train") });
    $("eng").value = "two"; $("eng").onchange({ target: $("eng") });
    $("train").onclick({ target: $("train") });
    while (e2.ep < ${+twoEpochsArg}) { TH = 0.5 + 2.5 * S(); step(); }
    let w2s = 0; for (let i = 0; i < W2.length; i++) w2s += W2[i];
    out.two = { tick, seed: seed >>> 0, em, epoch, w2Sum: w2s, log: logs(),
      talk: ["hi", "what is two plus three", "zzz"].map((q) => text2(score2(ctx2(q)))) };
    return out;`,
  curriculum: `setTM("cur2"); $("train").onclick({ target: $("train") }); run(4000); return state();`,
  response: `setTM("resp"); $("train").onclick({ target: $("train") }); run(3000); return state();`,
  language: `setTM("lang"); $("train").onclick({ target: $("train") }); run(3000); return state();`,
  focusAll: `$("focus").value = "all"; $("focus").onchange({ target: $("focus") });
    $("train").onclick({ target: $("train") }); run(3000); return state();`,
  generated: `$("src").value = "gen"; $("train").onclick({ target: $("train") }); run(3000); return state();`,
  noise: `$("no").value = "30"; $("no").oninput({ target: $("no") }); $("ip").value = "0"; $("ip").oninput({ target: $("ip") });
    $("spon").checked = false; $("spon").onchange({ target: $("spon") }); $("smart").checked = false; $("smart").onchange({ target: $("smart") });
    $("wait").checked = true; $("train").onclick({ target: $("train") }); run(2500); return state();`,
  eval: `$("ds").value = "hi => hello\\nbye => goodbye\\nyes => ok"; $("load").onclick(); $("train").onclick({ target: $("train") }); run(2000);
    $("train").onclick({ target: $("train") }); $("tlog").children.length = 0; $("eval").onclick();
    let guard = 0; while (mode === "eval" && guard++ < 20000) { TH = 0.5 + 2.5 * S(); step(); }
    return { ...state(), eOK, eN };`,
  talk: `
    const lines = []; let lastStr = "", stable = 0, lastLogged = "";
    const say = (t) => { codes = [...t.slice(0, 100)].map((c) => c.charCodeAt(0) & 255); stamp = (stamp % 65535) + 1; accS.fill(0); txtTTL = 20;
      for (let f = 0; f < 100; f++) { run(20); const s = readoutShown();
        if (s === lastStr) stable++; else { stable = 0; lastStr = s; }
        if (stable === 15 && s.trim() && s.trim() !== lastLogged) { lastLogged = s.trim(); lines.push("t" + tick + ": " + lastLogged); } }
      lines.push(t + " -> " + (readoutShown().trimEnd() || "(silent)") + " | bg " + (readout(acc).trim() || "-")); };
    say("hi"); say("what is two plus three");
    return { ...state(), lines };`,
  // Writes the page's export after some training to $EXPORT_TO (for the cross-load check).
  exportFile: `$("train").onclick({ target: $("train") }); run(1500);
    require("node:fs").writeFileSync(process.env.EXPORT_TO, JSON.stringify(snapshot())); return { tick };`,
  // Loads $IMPORT_FROM with the page's restore() and lists fields whose re-export differs.
  importFile: `const m = JSON.parse(require("node:fs").readFileSync(process.env.IMPORT_FROM, "utf8")); restore(m);
    const again = snapshot(); const bad = [];
    for (const k of Object.keys(again)) if (JSON.stringify(again[k]) !== JSON.stringify(m[k])) bad.push(k);
    return { tick, differ: bad };`,
  snapshot: `const m = snapshot(); const o = {};
    for (const [k, v] of Object.entries(m)) o[k] = typeof v === "string" && v.length > 200 ? { fnv: fnv(v), len: v.length } : v;
    return o;`,
};

const out = {};
const only = process.argv[5];
globalThis.require = (await import("node:module")).createRequire(import.meta.url);
for (const [name, code] of Object.entries(scenarios)) {
  if (only ? name !== only : name.endsWith("File")) continue;
  els.clear();
  out[name] = new Function(script + common + "\n;return (() => {" + code + "})();")();
}
console.log(JSON.stringify(out));
