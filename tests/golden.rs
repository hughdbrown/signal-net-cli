//! Runs the scenarios of `tools/js-golden.mjs` and compares with what the page's own code printed
//! (tests/fixtures/golden.json). Every scenario starts from a freshly loaded page.

// Test helpers outside #[test] functions: a panic with a line number is the right failure here.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use serde_json::Value;
use signal_net::{Engine, Focus, Lab, Settings, Source, TrainMode, format_pairs, parse_pairs};

fn fixture(name: &str) -> Value {
    let all: Value = serde_json::from_str(include_str!("fixtures/golden.json")).unwrap();
    all[name].clone()
}

fn close(a: f64, b: &Value, what: &str) {
    let b = b.as_f64().unwrap();
    assert!(
        (a - b).abs() <= 1e-9 * b.abs().max(1.0),
        "{what}: rust {a}, js {b}"
    );
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_owned())
        .collect()
}

fn log(lab: &mut Lab) -> Vec<String> {
    lab.drain_events().into_iter().map(|e| e.text).collect()
}

fn with(lab: &mut Lab, f: impl FnOnce(&mut Settings)) {
    let mut s = lab.settings().clone();
    f(&mut s);
    lab.set_settings(s);
}

fn run(lab: &mut Lab, ticks: usize) {
    for _ in 0..ticks {
        lab.step();
    }
}

/// Compares the page's `state()` record (and its training log) with the lab.
fn assert_state(lab: &mut Lab, s: &Value, log_lines: &[String]) {
    assert_eq!(log_lines, strings(&s["log"]).as_slice(), "training log");
    let c = lab.checksums();
    assert_eq!(u64::from(lab.tick()), s["tick"].as_u64().unwrap(), "tick");
    assert_eq!(
        lab.live_signals() as u64,
        s["ca"].as_u64().unwrap(),
        "live signals"
    );
    assert_eq!(u64::from(lab.seed()), s["seed"].as_u64().unwrap(), "seed");
    close(lab.exact_avg(), &s["em"], "em");
    assert_eq!(lab.epoch(), s["epoch"].as_u64().unwrap(), "epoch");
    assert_eq!(c.cur_pair as u64, s["cur"].as_u64().unwrap(), "cur");
    assert_eq!(u64::from(c.streak), s["streak"].as_u64().unwrap(), "streak");
    assert_eq!(c.pt, s["pt"].as_i64().unwrap(), "pt");
    close(c.sum_w, &s["sumW"], "sum of weights");
    assert_eq!(
        u64::from(c.sum_tag),
        s["sumTag"].as_u64().unwrap(),
        "sum of tags"
    );
    assert_eq!(
        u64::from(c.sum_tgt),
        s["sumTgt"].as_u64().unwrap(),
        "sum of targets"
    );
    close(c.sum_acc_s, &s["accSum"], "accS");
    assert_eq!(c.spawn_tot, s["spawnTot"].as_u64().unwrap(), "spawned");
    assert_eq!(c.drained, s["drT"].as_u64().unwrap(), "drained");
    assert_eq!(c.cap_hits, s["capHits"].as_u64().unwrap(), "ceiling hits");
    assert_eq!(c.smart_fixes, s["smFix"].as_u64().unwrap(), "smart fixes");
    assert_eq!(
        c.smart_penalties,
        s["smPen"].as_u64().unwrap(),
        "smart penalties"
    );
}

fn train_scenario(name: &str, ticks: usize, setup: impl FnOnce(&mut Settings)) {
    let g = fixture(name);
    let mut lab = Lab::browser_startup(Settings::default());
    with(&mut lab, setup);
    lab.start_training().unwrap();
    run(&mut lab, ticks);
    let l = log(&mut lab);
    assert_state(&mut lab, &g, &l);
}

#[test]
fn basic_signal_then_two_stage() {
    let g = fixture("basic");
    let mut lab = Lab::browser_startup(Settings::default());

    let s = &g["startup"];
    assert_eq!(
        format_pairs(lab.pair_list()),
        strings(&s["pairs"]).join("\n")
    );
    assert_eq!(
        u64::from(lab.seed()),
        s["seed"].as_u64().unwrap(),
        "startup seed"
    );
    let c = lab.checksums();
    close(c.sum_w, &s["sumW"], "startup sum of weights");
    assert_eq!(u64::from(c.sum_tag), s["sumTag"].as_u64().unwrap());
    assert_eq!(u64::from(c.sum_tgt), s["sumTgt"].as_u64().unwrap());

    let s = &g["sig"];
    lab.start_training().unwrap();
    run(&mut lab, s["ticks"].as_u64().unwrap() as usize);
    let l = log(&mut lab);
    assert_state(&mut lab, s, &l);

    let s = &g["two"];
    lab.stop();
    with(&mut lab, |s| s.engine = Engine::TwoStage);
    lab.start_training().unwrap();
    while lab.epoch() < 3 {
        lab.step();
    }
    assert_eq!(log(&mut lab), strings(&s["log"]));
    assert_eq!(u64::from(lab.tick()), s["tick"].as_u64().unwrap());
    assert_eq!(u64::from(lab.seed()), s["seed"].as_u64().unwrap());
    close(lab.exact_avg(), &s["em"], "two-stage em");
    close(lab.checksums().w2_sum, &s["w2Sum"], "readout weight sum");
    let talk: Vec<String> = ["hi", "what is two plus three", "zzz"]
        .iter()
        .map(|q| lab.answer_two_stage(q))
        .collect();
    assert_eq!(talk, strings(&s["talk"]));
}

#[test]
fn curriculum_mode() {
    train_scenario("curriculum", 4000, |s| s.train_mode = TrainMode::Curriculum);
}

#[test]
fn response_mode() {
    train_scenario("response", 3000, |s| s.train_mode = TrainMode::Response);
}

#[test]
fn language_mode() {
    train_scenario("language", 3000, |s| s.train_mode = TrainMode::Language);
}

#[test]
fn focus_all_pairs() {
    train_scenario("focusAll", 3000, |s| s.focus = Focus::All);
}

#[test]
fn generated_source() {
    train_scenario("generated", 3000, |s| s.source = Source::Generated);
}

#[test]
fn noise_without_spawning_or_smart_repair() {
    train_scenario("noise", 2500, |s| {
        s.noise = 30;
        s.interaction = 0;
        s.spawn = false;
        s.smart = false;
        s.wait_all = true;
    });
}

#[test]
fn evaluation_pass() {
    let g = fixture("eval");
    let mut lab = Lab::browser_startup(Settings::default());
    lab.load_pairs(parse_pairs("hi => hello\nbye => goodbye\nyes => ok"));
    lab.start_training().unwrap();
    run(&mut lab, 2000);
    lab.stop();
    lab.drain_events();
    lab.start_eval().unwrap();
    let mut guard = 0;
    while lab.mode() == signal_net::Mode::Eval && guard < 20_000 {
        lab.step();
        guard += 1;
    }
    let mut l = log(&mut lab);
    // The page's harness clears its log before evaluating, but the "EVAL:" line comes after.
    let summary = l.pop().unwrap();
    let (ok, n) = lab.eval_result();
    assert_eq!(summary, format!("EVAL: {ok}/{n} exact"));
    assert_eq!(ok, g["eOK"].as_u64().unwrap());
    assert_eq!(n, g["eN"].as_u64().unwrap());
    let mut want = strings(&g["log"]);
    assert_eq!(want.pop().unwrap(), summary);
    let g_without_summary = {
        let mut g = g.clone();
        g["log"] = Value::from(want);
        g
    };
    assert_state(&mut lab, &g_without_summary, &l);
}

#[test]
fn talking_to_the_signal_network() {
    let g = fixture("talk");
    let mut lab = Lab::browser_startup(Settings::default());
    let mut lines = Vec::new();
    for t in ["hi", "what is two plus three"] {
        lab.send_text(t);
        for _ in 0..100 {
            if let Some(line) = lab.talk_frame() {
                lines.push(line);
            }
        }
        let out = lab.shown_output();
        let bg = lab.background_output();
        lines.push(format!(
            "{t} -> {} | bg {}",
            if out.is_empty() { "(silent)" } else { &out },
            if bg.is_empty() { "-" } else { &bg }
        ));
    }
    assert_eq!(lines, strings(&g["lines"]));
    assert_state(&mut lab, &g, &[]);
}

fn fnv(s: &str) -> u64 {
    let mut h: u32 = 2_166_136_261;
    for u in s.encode_utf16() {
        h ^= u32::from(u);
        h = h.wrapping_mul(16_777_619);
    }
    u64::from(h)
}

fn same_json(rust: &Value, js: &Value, key: &str) {
    match (rust, js) {
        (Value::String(r), Value::Object(o)) if o.contains_key("fnv") => {
            assert_eq!(
                r.encode_utf16().count() as u64,
                o["len"].as_u64().unwrap(),
                "{key} length"
            );
            assert_eq!(fnv(r), o["fnv"].as_u64().unwrap(), "{key} contents");
        }
        (Value::Number(r), Value::Number(j)) => {
            assert!(
                (r.as_f64().unwrap() - j.as_f64().unwrap()).abs() < 1e-12,
                "{key}: {r} vs {j}"
            );
        }
        (Value::Array(r), Value::Array(j)) => {
            assert_eq!(r.len(), j.len(), "{key} length");
            for (i, (a, b)) in r.iter().zip(j).enumerate() {
                same_json(a, b, &format!("{key}[{i}]"));
            }
        }
        (Value::Object(r), Value::Object(j)) => {
            for (k, b) in j {
                same_json(r.get(k).unwrap_or(&Value::Null), b, &format!("{key}.{k}"));
            }
        }
        _ => assert_eq!(rust, js, "{key}"),
    }
}

#[test]
fn model_file_matches_the_page_export() {
    let g = fixture("snapshot");
    let lab = Lab::browser_startup(Settings::default());
    let m = lab.snapshot();
    let ours: Vec<&String> = m
        .as_object()
        .unwrap()
        .keys()
        .filter(|k| !k.starts_with("x_"))
        .collect();
    let theirs: Vec<&String> = g.as_object().unwrap().keys().collect();
    let mut a = ours.clone();
    let mut b = theirs.clone();
    a.sort();
    b.sort();
    assert_eq!(a, b, "same fields as the page's export");
    same_json(&m, &g, "model");
}

#[test]
fn model_file_round_trips() {
    let mut lab = Lab::browser_startup(Settings::default());
    lab.start_training().unwrap();
    run(&mut lab, 1200);
    let mut s = lab.settings().clone();
    s.engine = Engine::TwoStage;
    lab.stop();
    lab.set_settings(s);
    lab.start_training().unwrap();
    run(&mut lab, 200);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.json");
    signal_net::save_model(&lab, &path).unwrap();

    let mut back = Lab::new(Settings::default(), 1);
    signal_net::load_model(&mut back, &path).unwrap();
    let (a, b) = (back.snapshot(), lab.snapshot());
    let differ: Vec<&String> = b
        .as_object()
        .unwrap()
        .iter()
        .filter(|(k, v)| a.get(k.as_str()) != Some(v))
        .map(|(k, _)| k)
        .collect();
    assert!(
        differ.is_empty(),
        "fields that changed in the round trip: {differ:?}"
    );
    assert!(back.has_readout());
}

#[test]
fn bad_model_files_are_rejected_with_a_reason() {
    let mut lab = Lab::new(Settings::default(), 1);
    let mut m = lab.snapshot();
    m["tgt"] = Value::from("AAAA/w=="); // one target of -16777216
    let err = lab.restore(&m).unwrap_err();
    assert!(format!("{err:#}").contains("tgt"), "{err:#}");

    let mut m = Lab::new(Settings::default(), 1).snapshot();
    m["K"] = Value::from(5);
    let err = lab.restore(&m).unwrap_err();
    assert!(format!("{err:#}").contains("K is 5"), "{err:#}");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("broken.json");
    std::fs::write(&path, "{ not json").unwrap();
    let err = signal_net::load_model(&mut lab, &path).unwrap_err();
    assert!(format!("{err:#}").contains("parsing"), "{err:#}");
}

/// Loads a model the page exported and checks that saving it again changes nothing the page wrote.
/// Run with `SIGNAL_NET_BROWSER_EXPORT=file.json cargo test -- --ignored` (see tools/js-golden.mjs).
#[test]
#[ignore = "needs a browser export; see tools/js-golden.mjs exportFile"]
fn browser_export_loads_unchanged() {
    let path = std::env::var("SIGNAL_NET_BROWSER_EXPORT").expect("set SIGNAL_NET_BROWSER_EXPORT");
    let js: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let mut lab = Lab::new(Settings::default(), 1);
    lab.restore(&js).unwrap();
    let ours = lab.snapshot();
    let differ: Vec<&String> = js
        .as_object()
        .unwrap()
        .iter()
        .filter(|(k, v)| {
            let o = &ours[k.as_str()];
            match (o, v) {
                (Value::Number(a), Value::Number(b)) => a.as_f64() != b.as_f64(),
                _ => o != *v,
            }
        })
        .map(|(k, _)| k)
        .collect();
    assert!(differ.is_empty(), "fields that changed: {differ:?}");
    if let Ok(out) = std::env::var("SIGNAL_NET_SAVE_TO") {
        signal_net::save_model(&lab, std::path::Path::new(&out)).unwrap();
    }
}
