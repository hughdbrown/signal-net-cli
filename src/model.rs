//! Model files in the page's `signal-net-v4` JSON format.
//!
//! Every typed array is stored as base64 of its little-endian bytes, exactly as the page's "Export
//! model" writes it, so files move freely between the browser and this program. The program also
//! writes a few `x_` fields the page does not save (link guards, the struggle ledger and the
//! two-stage readout); the page ignores them, and files without them still load here.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use serde_json::{Map, Value, json};

use crate::lab::{D, Engine, Focus, K, L, Lab, MAXS, Mode, N, P, Stuck, TI, TO, TrainMode, V};
use crate::pairs::{Pair, format_pairs, parse_pairs};
use crate::two_stage::C2;

const FORMAT: &str = "signal-net-v4";

/// Plain numbers that are stored as little-endian bytes.
trait Pod: Copy {
    const SIZE: usize;
    fn put(self, out: &mut Vec<u8>);
    fn get(b: &[u8]) -> Self;
}

macro_rules! pod {
    ($($t:ty),*) => {$(
        impl Pod for $t {
            const SIZE: usize = std::mem::size_of::<$t>();
            fn put(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
            fn get(b: &[u8]) -> Self {
                let mut a = [0u8; std::mem::size_of::<$t>()];
                a.copy_from_slice(b);
                <$t>::from_le_bytes(a)
            }
        }
    )*};
}
pod!(u8, u16, i32, u32, f32);

fn enc<T: Pod>(v: &[T]) -> Value {
    let mut bytes = Vec::with_capacity(v.len() * T::SIZE);
    for &x in v {
        x.put(&mut bytes);
    }
    Value::String(STANDARD.encode(bytes))
}

/// Decodes a base64 array into the start of `dst` (like the page's `setArr`); returns the count.
fn dec_into<T: Pod>(dst: &mut [T], m: &Value, key: &str) -> Result<usize> {
    let s = m
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("field `{key}` is missing or not a string"))?;
    let bytes = STANDARD
        .decode(s)
        .with_context(|| format!("field `{key}` is not valid base64"))?;
    ensure!(
        bytes.len() % T::SIZE == 0,
        "field `{key}` has {} bytes, not a whole number of {}-byte values",
        bytes.len(),
        T::SIZE
    );
    let n = bytes.len() / T::SIZE;
    ensure!(
        n <= dst.len(),
        "field `{key}` holds {n} values but the network has room for {}",
        dst.len()
    );
    for (d, chunk) in dst.iter_mut().zip(bytes.chunks_exact(T::SIZE)) {
        *d = T::get(chunk);
    }
    Ok(n)
}

fn num(m: &Value, key: &str) -> Result<f64> {
    match m.get(key) {
        Some(Value::Number(n)) => n
            .as_f64()
            .with_context(|| format!("field `{key}` is not a usable number")),
        Some(Value::Null) => Ok(0.0), // JSON.stringify writes NaN as null
        Some(Value::String(s)) => s
            .trim()
            .parse()
            .with_context(|| format!("field `{key}` is not a number: {s:?}")),
        _ => bail!("field `{key}` is missing or not a number"),
    }
}

fn opt_num(m: &Value, key: &str) -> Result<Option<f64>> {
    match m.get(key) {
        None => Ok(None),
        Some(_) => num(m, key).map(Some),
    }
}

/// A non-negative integer field.
fn count(m: &Value, key: &str) -> Result<u64> {
    let x = num(m, key)?;
    ensure!(
        x >= 0.0 && x.fract() == 0.0 && x < 9e15,
        "field `{key}` should be a non-negative integer, got {x}"
    );
    Ok(x as u64)
}

fn index(m: &Value, key: &str, below: usize) -> Result<usize> {
    let x = count(m, key)? as usize;
    ensure!(
        x < below.max(1),
        "field `{key}` is {x}, expected below {below}"
    );
    Ok(x)
}

fn text<'a>(m: &'a Value, key: &str) -> Result<&'a str> {
    m.get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("field `{key}` is missing or not a string"))
}

fn array<'a>(m: &'a Value, key: &str) -> Result<&'a Vec<Value>> {
    m.get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("field `{key}` is missing or not a list"))
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|x| x != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
        Value::Null => false,
    }
}

impl Lab {
    /// The complete state as a model-file JSON value (the page's `snapshot()`).
    #[must_use]
    pub fn snapshot(&self) -> Value {
        let ca = self.ca;
        let mut m = Map::new();
        let mut put = |k: &str, v: Value| {
            m.insert(k.to_owned(), v);
        };
        put("fmt", json!(FORMAT));
        for (k, v) in [
            ("P", P),
            ("TI", TI),
            ("TO", TO),
            ("V", V),
            ("D", D),
            ("K", K),
            ("L", L),
        ] {
            put(k, json!(v));
        }
        put("ca", json!(ca));
        put("ds", json!(format_pairs(&self.list)));
        put(
            "ui",
            json!({
                "no": self.settings.noise.to_string(),
                "tp": self.settings.ticks_per_frame.to_string(),
                "ip": self.settings.interaction.to_string(),
                "hc": self.settings.hard_cap.to_string(),
                "tmode": self.settings.train_mode.js_name(),
                "adapt": self.settings.adaptive,
            }),
        );
        // The page keeps its seed as a signed 32-bit number.
        put("seed", json!(self.rng.seed() as i32));
        put("tick", json!(self.tick));
        put("keep", json!(self.keep));
        put("mode", json!(self.mode.js_name()));
        put("learn", json!(self.learn));
        put("epoch", json!(self.epoch));
        put("em", json!(self.em));
        put("ba", json!(self.ba));
        put("pt", json!(self.pt));
        put("ci", json!(self.ci));
        put("focus", json!(self.settings.focus.js_name()));
        put("cur", json!(self.cur_pair));
        put("streak", json!(self.streak));
        put("ptr", json!(self.ptr));
        put("eOK", json!(self.e_ok));
        put("eN", json!(self.e_n));
        put("txtTTL", json!(self.txt_ttl));
        put("visTTL", json!(self.vis_ttl));
        put("codes", json!(self.codes));
        put("tgtStr", json!(self.tgt_str));
        put("order", json!(self.order));
        put("last", json!(self.last));
        put("sel", json!(self.sel));
        put(
            "pairs",
            Value::Array(
                self.pairs
                    .iter()
                    .map(|p| json!([p.input, p.output]))
                    .collect(),
            ),
        );
        put("stamp", json!(self.stamp));
        put("sub", json!(1));
        put("cs", json!(self.stage));
        put("ta", json!(self.ta));
        put("va", json!(self.va));
        put("vn", json!(self.vn));
        put("stN", json!(self.st_n));
        put("br", enc(&self.br));
        put("rg", enc(&self.rg));
        put("rgi", json!(self.rgi));
        put("bt", enc(&self.bt));
        put("bq", enc(&self.bq));
        put("gl", json!(self.gl));
        put("ga", json!(self.ga));
        put("cons", enc(&self.cons));
        put("sE", enc(&self.cur.energy[..ca]));
        put("tgt", enc(&self.tgt));
        put("w", enc(&self.w));
        put("tag", enc(&self.tag));
        put("act", enc(&self.act));
        put("acc", enc(&self.acc));
        put("tb", enc(&self.tb));
        put("accS", enc(&self.acc_s));
        put("img", enc(&self.img));
        put("sn", enc(&self.cur.neuron[..ca]));
        put("sb", enc(&self.cur.birth[..ca]));
        put("se", enc(&self.cur.stamp[..ca]));
        put("sp", enc(&self.cur.payload[..ca]));
        put("ST", enc(&self.cur.trace[..ca * L]));
        // Extensions (ignored by the page).
        put("x_gk", enc(&self.gk));
        put("x_mlk", enc(&self.mlk));
        put("x_fm", enc(&self.fm));
        put("x_fx", enc(&self.fx));
        put(
            "x_eng",
            json!(match self.settings.engine {
                Engine::Signal => "sig",
                Engine::TwoStage => "two",
            }),
        );
        if let Some(w2) = &self.w2 {
            put("x_w2", enc(w2));
        }
        put("x_usd", enc(&self.usd));
        put(
            "x_state",
            json!({
                "tr": self.tr,
                "hr": self.hr,
                "rc": self.rc,
                "lrs": self.lrs,
                "mst": self.mst,
                "curTries": self.cur_tries,
                "e2ep": self.e2.ep,
                "trials": self.trials_done,
                "smFix": self.sm_fix,
                "smPen": self.sm_pen,
                "spawnTot": self.spawn_tot,
                "drT": self.dr_t,
                "capHits": self.cap_hits,
                "stk": self.stk.as_ref().map(|s| json!([s.pair.input, s.pair.output, s.streak])),
                "ca": self.ca_map,
                "rh": self.rh,
                "rt": self.rt,
            }),
        );
        Value::Object(m)
    }

    /// Restores a model-file value (the page's `restore()`), checking every index it will later use.
    ///
    /// # Errors
    /// When the file is not a compatible model or holds out-of-range values. The lab may then be
    /// partly overwritten and should be discarded.
    pub fn restore(&mut self, m: &Value) -> Result<()> {
        let fmt = text(m, "fmt")?;
        let v3 = fmt == "signal-net-v3";
        ensure!(
            fmt == FORMAT || v3,
            "not a compatible model file (format {fmt:?})"
        );
        for (k, want) in [
            ("P", P),
            ("TI", TI),
            ("TO", TO),
            ("V", V),
            ("K", K),
            ("L", L),
        ] {
            let got = count(m, k)?;
            ensure!(
                got == want as u64,
                "not a compatible model file: {k} is {got}, expected {want}"
            );
        }
        if !v3 {
            let got = count(m, "D")?;
            ensure!(
                got == D as u64,
                "not a compatible model file: D is {got}, expected {D}"
            );
        }
        ensure!(
            opt_num(m, "sub")? == Some(1.0),
            "not a compatible model file (no submit neuron); start a new network instead"
        );
        self.nt.fill(0);
        dec_into(&mut self.cons, m, "cons")?;
        dec_into(&mut self.tgt, m, "tgt")?;
        if let Some(bad) = self.tgt.iter().find(|&&t| t < 0 || t as usize >= N) {
            bail!("field `tgt` holds link target {bad}, outside 0..{N}");
        }
        dec_into(&mut self.w, m, "w")?;
        dec_into(&mut self.tag, m, "tag")?;
        self.ensure_sub();
        if v3 {
            self.seed_drains(crate::lab::DRP);
        }
        dec_into(&mut self.act, m, "act")?;
        dec_into(&mut self.acc, m, "acc")?;
        dec_into(&mut self.tb, m, "tb")?;
        dec_into(&mut self.img, m, "img")?;
        let ca = count(m, "ca")? as usize;
        ensure!(ca <= MAXS, "field `ca` is {ca}, more than {MAXS} signals");
        let checked = |n: usize, key: &str| -> Result<()> {
            ensure!(
                n == ca,
                "field `{key}` holds {n} signals but `ca` says {ca}"
            );
            Ok(())
        };
        checked(dec_into(&mut self.cur.neuron, m, "sn")?, "sn")?;
        if let Some(bad) = self.cur.neuron[..ca]
            .iter()
            .find(|&&n| n < 0 || n as usize >= N)
        {
            bail!("field `sn` holds neuron {bad}, outside 0..{N}");
        }
        checked(dec_into(&mut self.cur.energy, m, "sE")?, "sE")?;
        let tick = count(m, "tick")? as u32;
        if m.get("sb").is_some() {
            checked(dec_into(&mut self.cur.birth, m, "sb")?, "sb")?;
        } else {
            self.cur.birth[..ca].fill(tick);
        }
        checked(dec_into(&mut self.cur.stamp, m, "se")?, "se")?;
        checked(dec_into(&mut self.cur.payload, m, "sp")?, "sp")?;
        let st = dec_into(&mut self.cur.trace, m, "ST")?;
        ensure!(
            st == ca * L,
            "field `ST` holds {st} hops, expected {}",
            ca * L
        );
        let links = (N * K) as i32;
        if let Some(bad) = self.cur.trace[..ca * L]
            .iter()
            .find(|&&j| j < -1 || j >= links)
        {
            bail!("field `ST` holds link {bad}, outside -1..{links}");
        }
        self.ca = ca;
        self.na = 0;

        let seed = num(m, "seed")?;
        ensure!(
            seed.fract() == 0.0 && seed.abs() < 5e9,
            "field `seed` is not a 32-bit integer"
        );
        self.rng.set_seed((seed as i64) as u32);
        self.tick = tick;
        self.keep = num(m, "keep")?;
        let mode = text(m, "mode")?;
        self.mode = Mode::from_js(mode).with_context(|| format!("unknown mode {mode:?}"))?;
        self.learn = m.get("learn").is_some_and(truthy);
        self.epoch = count(m, "epoch")?;
        self.em = num(m, "em")?;
        self.ba = num(m, "ba")?;
        self.pt = count(m, "pt")? as i64;
        let focus = text(m, "focus")?;
        self.settings.focus =
            Focus::from_js(focus).with_context(|| format!("unknown focus {focus:?}"))?;
        self.streak = count(m, "streak")? as u32;
        self.e_ok = count(m, "eOK")?;
        self.e_n = count(m, "eN")?;
        self.txt_ttl = count(m, "txtTTL")? as u32;
        self.vis_ttl = count(m, "visTTL")? as u32;
        self.codes = array(m, "codes")?
            .iter()
            .map(|v| v.as_u64().filter(|&c| c < 256).map(|c| c as u8))
            .collect::<Option<Vec<u8>>>()
            .context("field `codes` must hold character codes 0..255")?;
        ensure!(
            self.codes.len() <= 100,
            "field `codes` holds more than 100 characters"
        );
        text(m, "tgtStr")?.clone_into(&mut self.tgt_str);
        self.tgt_units = self.tgt_str.encode_utf16().collect();
        self.pairs = array(m, "pairs")?
            .iter()
            .map(|p| {
                let a = p.get(0).and_then(Value::as_str)?;
                let b = p.get(1).and_then(Value::as_str)?;
                Some(Pair::new(a, b))
            })
            .collect::<Option<Vec<_>>>()
            .context("field `pairs` must be a list of [input, response] strings")?;
        let np = self.pairs.len();
        self.ci = index(m, "ci", np)?;
        self.cur_pair = index(m, "cur", np)?;
        self.order = array(m, "order")?
            .iter()
            .map(|v| v.as_u64().map(|i| i as usize).filter(|&i| i < np))
            .collect::<Option<Vec<_>>>()
            .context("field `order` must hold pair numbers")?;
        self.ptr = (count(m, "ptr")? as usize).min(self.order.len());
        let last = array(m, "last")?;
        self.last = (0..np)
            .map(|i| {
                last.get(i)
                    .and_then(Value::as_i64)
                    .map_or(-1, |x| x.clamp(-1, 1) as i8)
            })
            .collect();
        let sel = array(m, "sel")?;
        self.sel = (0..np).map(|i| sel.get(i).is_some_and(truthy)).collect();
        self.list = parse_pairs(text(m, "ds")?);

        let stamp = num(m, "stamp")?;
        self.stamp = if (1.0..=65_535.0).contains(&stamp) {
            stamp as u16
        } else {
            1
        };
        self.ft.fill(0);
        let ui = m.get("ui").context("field `ui` is missing")?;
        if let Some(x) = opt_num(ui, "no")? {
            self.settings.noise = (x.max(0.0) as usize).min(500);
        }
        if let Some(x) = opt_num(ui, "tp")? {
            self.settings.ticks_per_frame = (x.max(1.0) as usize).min(20);
        }
        if let Some(x) = opt_num(ui, "ip")? {
            self.settings.interaction = x.clamp(0.0, 100.0) as u32;
        }
        if let Some(x) = opt_num(ui, "hc")? {
            self.settings.hard_cap = (x.max(1.0) as usize).min(MAXS);
        }
        if let Some(t) = ui
            .get("tmode")
            .and_then(Value::as_str)
            .and_then(TrainMode::from_js)
        {
            self.settings.train_mode = t;
        }
        if let Some(a) = ui.get("adapt") {
            self.settings.adaptive = truthy(a);
        }
        if m.get("accS").is_some() {
            dec_into(&mut self.acc_s, m, "accS")?;
        }
        if m.get("cs").is_some() {
            self.stage = count(m, "cs")? as usize;
            self.ta = num(m, "ta")?;
            self.va = num(m, "va")?;
            self.vn = num(m, "vn")?;
            self.st_n = count(m, "stN")?;
        }
        if m.get("br").is_some() {
            dec_into(&mut self.br, m, "br")?;
        }
        if m.get("rg").is_some() {
            dec_into(&mut self.rg, m, "rg")?;
            if let Some(bad) = self.rg.iter().find(|&&n| n < 0 || n as usize >= N) {
                bail!("field `rg` holds neuron {bad}, outside 0..{N}");
            }
            self.rgi = count(m, "rgi")? as usize;
            dec_into(&mut self.bt, m, "bt")?;
            dec_into(&mut self.bq, m, "bq")?;
            self.gl = num(m, "gl")?.clamp(0.0, 3.0) as i32;
            self.ga = num(m, "ga")?;
        }
        for (key, dst) in [
            ("x_gk", &mut self.gk),
            ("x_mlk", &mut self.mlk),
            ("x_fm", &mut self.fm),
            ("x_fx", &mut self.fx),
        ] {
            if m.get(key).is_some() {
                dec_into(dst, m, key)?;
            }
        }
        match m.get("x_eng").and_then(Value::as_str) {
            Some("two") => self.settings.engine = Engine::TwoStage,
            Some("sig") => self.settings.engine = Engine::Signal,
            _ => {}
        }
        if m.get("x_w2").is_some() {
            let mut w2 = vec![0.0f32; C2 * TO];
            let n = dec_into(&mut w2, m, "x_w2")?;
            ensure!(
                n == C2 * TO,
                "field `x_w2` holds {n} weights, expected {}",
                C2 * TO
            );
            self.w2 = Some(w2);
        }
        if m.get("x_usd").is_some() {
            dec_into(&mut self.usd, m, "x_usd")?;
        }
        if let Some(x) = m.get("x_state") {
            self.restore_extra(x).context("field `x_state`")?;
        }
        Ok(())
    }

    /// Learning state carried between trials that the page does not export.
    fn restore_extra(&mut self, x: &Value) -> Result<()> {
        self.tr = num(x, "tr")?.clamp(100.0, 4000.0);
        self.hr = num(x, "hr")?;
        self.rc = num(x, "rc")?;
        self.lrs = num(x, "lrs")?;
        self.mst = count(x, "mst")? as usize;
        self.cur_tries = count(x, "curTries")? as u32;
        self.e2.ep = count(x, "e2ep")?;
        self.trials_done = count(x, "trials")?;
        self.sm_fix = count(x, "smFix")?;
        self.sm_pen = count(x, "smPen")?;
        self.spawn_tot = count(x, "spawnTot")?;
        self.dr_t = count(x, "drT")?;
        self.cap_hits = count(x, "capHits")?;
        self.stk = match x.get("stk") {
            Some(Value::Array(a)) => {
                let pair = a
                    .first()
                    .and_then(Value::as_str)
                    .zip(a.get(1).and_then(Value::as_str))
                    .context("`stk` must be [input, response, streak]")?;
                let streak = a.get(2).and_then(Value::as_u64).context("`stk` streak")?;
                Some(Stuck {
                    pair: Pair::new(pair.0, pair.1),
                    streak: streak as u32,
                })
            }
            _ => None,
        };
        self.ca_map = x
            .get("ca")
            .and_then(Value::as_object)
            .context("`ca` must be an object")?
            .iter()
            .map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
            .collect::<Option<_>>()
            .context("`ca` values must be numbers")?;
        let list = |key: &str| -> Result<std::collections::VecDeque<String>> {
            array(x, key)?
                .iter()
                .map(|v| v.as_str().map(str::to_owned))
                .collect::<Option<_>>()
                .with_context(|| format!("`{key}` must be a list of strings"))
        };
        self.rh = list("rh")?;
        self.rt = list("rt")?;
        ensure!(
            self.rh.len() == self.rt.len(),
            "`rh` and `rt` differ in length"
        );
        Ok(())
    }
}

/// Writes the lab to a model file.
///
/// # Errors
/// When the file cannot be written.
pub fn save_model(lab: &Lab, path: &Path) -> Result<()> {
    let file = fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut out = BufWriter::new(file);
    serde_json::to_writer(&mut out, &lab.snapshot())
        .with_context(|| format!("writing {}", path.display()))?;
    out.flush()
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Loads a model file into the lab.
///
/// # Errors
/// When the file cannot be read or is not a compatible model.
pub fn load_model(lab: &mut Lab, path: &Path) -> Result<()> {
    let data = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let m: Value = serde_json::from_str(&data)
        .with_context(|| format!("parsing {} as JSON", path.display()))?;
    lab.restore(&m)
        .with_context(|| format!("loading model {}", path.display()))
}
