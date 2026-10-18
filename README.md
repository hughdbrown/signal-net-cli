# signal-net

A command-line port of **Signal Network Lab v3.1** (`Signal_Network_Lab_3_11_4 9.TWO-STAGE.html`),
with both learning engines:

- **signal** — the original network: 500,000 processing neurons, signals that carry their question's
  payload along tag-matched weighted links, path credit at the output bits, drains, spawning, link
  rewiring, smart connecting, struggle ledger, all four training modes and the curriculum.
- **two-stage** — sparse context layer + error-correcting readout (fix 6).

The port follows the JavaScript step for step: the same xorshift stream, the same float32 storage,
the same order of every random draw. `tests/golden.rs` replays ten scenarios against values printed
by the page's own code (`tools/js-golden.mjs`) and they match exactly, including every line of the
training log. Model files use the page's `signal-net-v4` export format, so you can train here and
import in the browser, or the other way round.

## Usage

```sh
cargo build --release
B=target/release/signal-net

$B pairs                                   # the page's generated 131-pair list
$B train --engine two-stage --epochs 75 --save two.json
$B eval  --load two.json                   # EVAL: 131/131 exact (two-stage)
$B talk  --load two.json "what is seven plus seven" hey
$B train --trials 500 --save sig.json -v   # signal engine, every graded response printed
$B train --load sig.json --mode curriculum # resume; Ctrl-C stops cleanly (and saves with --save)
$B talk  --load sig.json                   # interactive: one message per line
```

`train` runs until `--epochs`, `--trials` or `--ticks` is reached, or until Ctrl-C. Every page
control is an option (`--mode`, `--focus`, `--source`, `--noise`, `--interaction`, `--hard-cap`,
`--spawn-*`, `--no-smart`, `--wait-all`, ...); `signal-net train --help` lists them. Options given on
the command line override the settings stored in a loaded model.

Pair files use the page's format, one `input => response` per line.

## Differences from the page

- **Not ported:** the Claude trainer (it needs the browser artifact's `claude.use()` API), vision
  input (the page marks it "not implemented yet"), the heat-map and I/O canvases, and ticking
  checkboxes in the pair list (`--focus checked` uses the selection stored in a browser model).
- **Model files carry more state.** The page's export omits the link guards (`gk`, `mlk`), the
  struggle ledger, link-usage flags, the transit estimate, the curriculum's memory and the two-stage
  readout. This program also writes them as `x_*` fields so a saved run resumes where it stopped.
  The page ignores those fields; files without them load fine.
- **Display does not draw random numbers.** On the page, every animation frame samples 20,000 random
  weights for the "mean weight" readout, and frames are cut off after 20 ms, so results depend on
  frame rate. Here the status line uses a fixed sample. Only the page's first frame at load time is
  reproduced (so a fresh run matches a freshly loaded page).
- A fresh process always starts from a clean network, so the page's Reset quirk (link guards
  surviving a reset) does not arise.

## Development

```sh
cargo test                                  # unit tests + golden scenarios (~25 s)
node tools/js-golden.mjs "../Signal_Network_Lab_3_11_4 9.TWO-STAGE.html" > tests/fixtures/golden.json
```

Regenerate the fixture after changing the HTML. To check real model files in both directions:

```sh
EXPORT_TO=/tmp/js.json node tools/js-golden.mjs "../Signal_Network_Lab_3_11_4 9.TWO-STAGE.html" 0 0 exportFile
SIGNAL_NET_BROWSER_EXPORT=/tmp/js.json SIGNAL_NET_SAVE_TO=/tmp/rs.json cargo test -- --ignored
IMPORT_FROM=/tmp/rs.json node tools/js-golden.mjs "../Signal_Network_Lab_3_11_4 9.TWO-STAGE.html" 0 0 importFile
```

`.cargo/config.toml` links with `ld64.lld` (`brew install lld`); delete it if lld is not installed.
