//! The lab: network constants, settings, and the state shared by both learning engines.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Write as _;

use crate::corpus::{Lm, generated_list};
use crate::pairs::{Pair, format_pairs, parse_pairs};
use crate::rng::Rng;

/// Grid width of the processing layer.
pub const GW: usize = 1000;
/// Grid height of the processing layer.
pub const GH: usize = 500;
pub(crate) const OX0: usize = GW - 20;
pub(crate) const OY0: usize = GH - 40;
/// Processing neurons.
pub const P: usize = GW * GH;
/// Text-input bit neurons (100 slots x 8 bits).
pub const TI: usize = 800;
/// Text-output bit neurons (100 slots x 8 bits).
pub const TO: usize = 800;
/// Vision-input neurons (100 x 100).
pub const V: usize = 10_000;
/// Drain neurons.
pub const D: usize = 800;
/// All neurons: processing, text in, text out, vision, submit, drains.
pub const N: usize = P + TI + TO + V + 1 + D;
/// Outgoing links per neuron.
pub const K: usize = 6;
/// Hops of path history a stimulus signal remembers.
pub const L: usize = 40;
pub(crate) const OI: usize = P;
pub(crate) const OO: usize = P + TI;
pub(crate) const OV: usize = P + TI + TO;
pub(crate) const OS: usize = OV + V;
pub(crate) const OD: usize = OS + 1;
pub(crate) const DRP: f64 = 0.003;
pub(crate) const SX: usize = GW - 12;
pub(crate) const SY: usize = GH - 72;
/// Most signals that can be alive at once.
pub const MAXS: usize = 300_000;
pub(crate) const PT: i64 = 1500;
pub(crate) const MINT: i64 = 150;
pub(crate) const NRESP: u32 = 5;
pub(crate) const RF: i64 = 100;
pub(crate) const ENF: u16 = 800;
pub(crate) const ENN: u16 = 150;
pub(crate) const HM: usize = 600;
pub(crate) const DM: usize = 300;
pub(crate) const GM: usize = 300;
pub(crate) const CH: f32 = 0.35;
pub(crate) const LAM: f64 = 2.0;
/// Consecutive trials whose response contains the answer before a pair counts as perfected.
pub(crate) const PERFECT: u32 = 7;
pub(crate) const PROMOTE: usize = 10;
/// The page's starting seed.
pub const DEFAULT_SEED: u32 = 123_456_789;

/// Which learning engine drives training, evaluation and talking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Engine {
    /// Signal network (original): signals travel weighted links and paths are credited.
    #[value(name = "signal")]
    Signal,
    /// Two-stage: sparse context layer + error-correcting readout.
    #[value(name = "two-stage")]
    TwoStage,
}

/// How graded responses are rewarded (the page's "Training mode").
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum TrainMode {
    /// 0. Lab-style curriculum: pretrain, SFT, RL (generates its own pairs).
    Curriculum,
    /// 1. Get a response at all.
    Response,
    /// 2. Meaningful results (match targets).
    Meaning,
    /// 3. Language (grammar + context, no fixed target).
    Language,
}

impl TrainMode {
    pub(crate) fn js_name(self) -> &'static str {
        match self {
            Self::Curriculum => "cur2",
            Self::Response => "resp",
            Self::Meaning => "meaning",
            Self::Language => "lang",
        }
    }

    pub(crate) fn from_js(s: &str) -> Option<Self> {
        [
            Self::Curriculum,
            Self::Response,
            Self::Meaning,
            Self::Language,
        ]
        .into_iter()
        .find(|m| m.js_name() == s)
    }
}

/// Which pairs are trained, in what order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Focus {
    /// All pairs, shuffled each round.
    All,
    /// Only pairs marked as checked (set in the browser and kept in model files).
    Checked,
    /// One at a time, until the answer comes out 7 trials in a row.
    OneAtATime,
}

impl Focus {
    pub(crate) fn js_name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Checked => "sel",
            Self::OneAtATime => "cur",
        }
    }

    pub(crate) fn from_js(s: &str) -> Option<Self> {
        [Self::All, Self::Checked, Self::OneAtATime]
            .into_iter()
            .find(|m| m.js_name() == s)
    }
}

/// Where training pairs come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Source {
    /// The pair list.
    List,
    /// An endless generated corpus whose level rises with the score.
    Generated,
}

/// What the lab is doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Idle,
    Train,
    Eval,
}

impl Mode {
    pub(crate) fn js_name(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Train => "train",
            Self::Eval => "eval",
        }
    }

    pub(crate) fn from_js(s: &str) -> Option<Self> {
        [Self::Idle, Self::Train, Self::Eval]
            .into_iter()
            .find(|m| m.js_name() == s)
    }
}

/// Every setting the page exposes as a control.
#[derive(Debug, Clone)]
pub struct Settings {
    pub engine: Engine,
    pub train_mode: TrainMode,
    pub focus: Focus,
    pub source: Source,
    /// Smart connecting: after a bad response, penalise links that lit wrong bits and add links toward
    /// bits that stayed dark.
    pub smart: bool,
    /// Signal structures spawn new signals.
    pub spawn: bool,
    /// Spawn chance per structure, 0..=1.
    pub spawn_rate: f64,
    /// Structure window, in ticks.
    pub spawn_window: u32,
    /// Live-signal count that spawning fades out toward.
    pub spawn_target: usize,
    /// Background noise signals per tick.
    pub noise: usize,
    /// Chance (percent) that two signals meeting at a neuron swap half their payload bits.
    pub interaction: u32,
    /// Hard ceiling on live signals.
    pub hard_cap: usize,
    /// Start each input bit with a strong link to the same output bit (applies when a network is built).
    pub identity_prior: bool,
    /// No distance limit on links (applies when links are created).
    pub unlimited_reach: bool,
    /// Grader tightens as exact match rises (language mode).
    pub adaptive: bool,
    /// Wait for all 5 responses (up to 6000 ticks) instead of cutting off at 1500.
    pub wait_all: bool,
    /// Ticks per display frame; used by `talk` to decide when the output has settled.
    pub ticks_per_frame: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            engine: Engine::Signal,
            train_mode: TrainMode::Meaning,
            focus: Focus::OneAtATime,
            source: Source::List,
            smart: true,
            spawn: true,
            spawn_rate: 0.3,
            spawn_window: 12,
            spawn_target: 40_000,
            noise: 0,
            interaction: 50,
            hard_cap: MAXS,
            identity_prior: true,
            unlimited_reach: true,
            adaptive: true,
            wait_all: false,
            ticks_per_frame: 20,
        }
    }
}

/// What a log line is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    /// One graded response (the page's training log).
    Response,
    /// Epoch/evaluation results, promotions and mode switches.
    Notice,
}

/// A line the page would have written to its training log.
#[derive(Debug, Clone)]
pub struct Event {
    pub kind: EventKind,
    pub text: String,
}

/// The live signals of one tick (the page keeps two of these and swaps them every tick).
#[derive(Debug)]
pub(crate) struct Signals {
    pub(crate) energy: Vec<u16>,
    pub(crate) birth: Vec<u32>,
    pub(crate) neuron: Vec<i32>,
    pub(crate) stamp: Vec<u16>,
    pub(crate) payload: Vec<u32>,
    pub(crate) trace: Vec<i32>,
}

impl Signals {
    fn new() -> Self {
        Self {
            energy: vec![0; MAXS],
            birth: vec![0; MAXS],
            neuron: vec![0; MAXS],
            stamp: vec![0; MAXS],
            payload: vec![0; MAXS],
            trace: vec![0; MAXS * L],
        }
    }
}

/// The pair the curriculum keeps training until it is perfected.
#[derive(Debug, Clone)]
pub(crate) struct Stuck {
    pub(crate) pair: Pair,
    pub(crate) streak: u32,
}

/// Mode-3 grading of one response.
#[derive(Debug, Clone)]
pub(crate) struct LangScore {
    pub(crate) r: f64,
    pub(crate) cs: [f32; 100],
}

/// Progress of the two-stage engine through one pass over the pairs.
#[derive(Debug, Clone)]
pub(crate) struct TwoPass {
    pub(crate) ord: Vec<usize>,
    pub(crate) p: usize,
    pub(crate) ex: usize,
    pub(crate) ep: u64,
    pub(crate) mode: Option<Mode>,
}

/// The whole simulation. Field names follow the JavaScript so the port can be read side by side.
#[derive(Debug)]
pub struct Lab {
    pub(crate) settings: Settings,
    pub(crate) rng: Rng,
    pub(crate) events: Vec<Event>,

    // Links: one entry per (neuron, k).
    pub(crate) usd: Vec<u8>,
    pub(crate) cons: Vec<f32>,
    pub(crate) tgt: Vec<i32>,
    pub(crate) w: Vec<f32>,
    pub(crate) tag: Vec<u32>,
    pub(crate) gk: Vec<f32>,
    pub(crate) mlk: Vec<f32>,
    pub(crate) rg: Vec<i32>,
    pub(crate) rgi: usize,
    pub(crate) ex_tab: [f32; 33],
    pub(crate) act: Vec<u8>,
    pub(crate) acc: Vec<f32>,
    pub(crate) acc_s: Vec<f32>,
    pub(crate) acc_b: Vec<f32>,
    pub(crate) acc_d: Vec<f32>,
    pub(crate) tb: Vec<u8>,
    pub(crate) img: Vec<f32>,

    // Signals.
    pub(crate) cur: Signals,
    pub(crate) next: Signals,
    pub(crate) ca: usize,
    pub(crate) na: usize,
    pub(crate) cap: usize,
    pub(crate) tr_buf: [i32; L],
    pub(crate) pk: [f64; K],
    pub(crate) cnt: Vec<u32>,
    pub(crate) ord: Vec<u32>,
    pub(crate) fi: Vec<u32>,
    pub(crate) ft: Vec<u32>,
    pub(crate) nt: Vec<u32>,
    pub(crate) np: Vec<u32>,
    pub(crate) nf: Vec<u16>,
    pub(crate) spawn_tot: u64,
    pub(crate) spawn_tick: u64,
    pub(crate) cap_hits: u64,
    pub(crate) dr_t: u64,
    pub(crate) stamp: u16,
    pub(crate) tick: u32,
    pub(crate) keep: f64,

    // Trial and grading state.
    pub(crate) mode: Mode,
    pub(crate) learn: bool,
    pub(crate) list: Vec<Pair>,
    pub(crate) pairs: Vec<Pair>,
    pub(crate) order: Vec<usize>,
    pub(crate) last: Vec<i8>,
    pub(crate) sel: Vec<bool>,
    pub(crate) ptr: usize,
    pub(crate) ci: usize,
    pub(crate) cur_pair: usize,
    pub(crate) streak: u32,
    pub(crate) cur_tries: u32,
    pub(crate) pt: i64,
    pub(crate) epoch: u64,
    pub(crate) em: f64,
    pub(crate) ba: f64,
    pub(crate) codes: Vec<u8>,
    pub(crate) tgt_str: String,
    pub(crate) tgt_units: Vec<u16>,
    pub(crate) e_ok: u64,
    pub(crate) e_n: u64,
    pub(crate) txt_ttl: u32,
    pub(crate) vis_ttl: u32,
    pub(crate) sub_hit: bool,
    pub(crate) sc: usize,
    pub(crate) first_hit: i64,
    pub(crate) tr: f64,
    pub(crate) hn: usize,
    pub(crate) ec: u32,
    pub(crate) nr: u32,
    pub(crate) last_sub: i64,
    pub(crate) trial_ex: bool,
    pub(crate) perfect_now: bool,
    pub(crate) trials_done: u64,
    pub(crate) last_reach: usize,
    pub(crate) rc: f64,
    pub(crate) hr: f64,
    pub(crate) ys: [u8; 100],
    pub(crate) hp: Vec<i32>,
    pub(crate) hc: Vec<u8>,
    pub(crate) hpp: Vec<u32>,
    pub(crate) hw: Vec<u8>,
    pub(crate) hb: Vec<u16>,
    pub(crate) spc: Vec<i32>,
    pub(crate) dp: Vec<i32>,
    pub(crate) dpp: Vec<u32>,
    pub(crate) dn: usize,
    pub(crate) gt: Vec<i32>,
    pub(crate) gn: usize,
    pub(crate) th: f64,
    pub(crate) ths: [f32; 100],
    pub(crate) fm: Vec<f32>,
    pub(crate) fx: Vec<f32>,
    pub(crate) struggle: String,
    pub(crate) sm_fix: u64,
    pub(crate) sm_pen: u64,
    pub(crate) len: usize,
    pub(crate) first_got: String,

    // Language grading.
    pub(crate) rs: f64,
    pub(crate) tw: HashSet<String>,
    pub(crate) lm: Option<Lm>,
    pub(crate) assoc: HashSet<String>,
    pub(crate) gl: i32,
    pub(crate) ga: f64,

    // Per-slot baselines and curriculum.
    pub(crate) bq: Vec<f32>,
    pub(crate) br: Vec<f32>,
    pub(crate) tk: [u8; 100],
    pub(crate) bt: [f32; 100],
    pub(crate) adv: [f32; 100],
    pub(crate) ca_map: HashMap<String, f64>,
    pub(crate) rh: VecDeque<String>,
    pub(crate) rt: VecDeque<String>,
    pub(crate) stk: Option<Stuck>,
    pub(crate) lrs: f64,
    pub(crate) stage: usize,
    pub(crate) ta: f64,
    pub(crate) va: f64,
    pub(crate) vn: f64,
    pub(crate) st_n: u64,
    pub(crate) mst: usize,

    // Talk-to-it display state.
    pub(crate) last_str: String,
    pub(crate) stable: u32,
    pub(crate) last_logged: String,

    // Two-stage engine.
    pub(crate) w2: Option<Vec<f32>>,
    pub(crate) z2: Vec<f32>,
    pub(crate) o2: String,
    pub(crate) e2: TwoPass,
}

impl Lab {
    /// Builds a fresh, randomly wired network (`init()` in the JavaScript). It has no pairs yet.
    #[must_use]
    pub fn new(settings: Settings, seed: u32) -> Self {
        let mut ex_tab = [0f32; 33];
        for (i, e) in ex_tab.iter_mut().enumerate() {
            *e = (7.0 * ((32.0 - i as f64) / 32.0 - 0.5)).exp() as f32;
        }
        let mut lab = Self {
            cap: settings.hard_cap,
            settings,
            rng: Rng::new(seed),
            events: Vec::new(),
            usd: vec![0; N * K],
            cons: vec![0.0; N * K],
            tgt: vec![0; N * K],
            w: vec![0.0; N * K],
            tag: vec![0; N * K],
            gk: vec![0.0; N * K],
            mlk: vec![0.0; N * K],
            rg: vec![0; 256],
            rgi: 0,
            ex_tab,
            act: vec![0; P],
            acc: vec![0.0; TO],
            acc_s: vec![0.0; TO],
            acc_b: vec![0.0; TO],
            acc_d: vec![0.0; TO],
            tb: vec![0; TO],
            img: vec![0.0; V],
            cur: Signals::new(),
            next: Signals::new(),
            ca: 0,
            na: 0,
            tr_buf: [0; L],
            pk: [0.0; K],
            cnt: vec![0; 1025],
            ord: vec![0; MAXS],
            fi: vec![0; P],
            ft: vec![0; P],
            nt: vec![0; P],
            np: vec![0; P],
            nf: vec![0; P],
            spawn_tot: 0,
            spawn_tick: 0,
            cap_hits: 0,
            dr_t: 0,
            stamp: 1,
            tick: 0,
            keep: 1.0,
            mode: Mode::Idle,
            learn: false,
            list: Vec::new(),
            pairs: Vec::new(),
            order: Vec::new(),
            last: Vec::new(),
            sel: Vec::new(),
            ptr: 0,
            ci: 0,
            cur_pair: 0,
            streak: 0,
            cur_tries: 0,
            pt: 0,
            epoch: 0,
            em: 0.0,
            ba: 0.0,
            codes: Vec::new(),
            tgt_str: String::new(),
            tgt_units: Vec::new(),
            e_ok: 0,
            e_n: 0,
            txt_ttl: 0,
            vis_ttl: 0,
            sub_hit: false,
            sc: 0,
            first_hit: -1,
            tr: 1500.0,
            hn: 0,
            ec: 0,
            nr: 0,
            last_sub: -1_000_000_000,
            trial_ex: false,
            perfect_now: false,
            trials_done: 0,
            last_reach: 0,
            rc: 0.0,
            hr: 0.0,
            ys: [0; 100],
            hp: vec![0; HM * L],
            hc: vec![0; HM],
            hpp: vec![0; HM],
            hw: vec![0; HM],
            hb: vec![0; HM],
            spc: vec![0; 8 * L],
            dp: vec![0; DM * L],
            dpp: vec![0; DM],
            dn: 0,
            gt: vec![0; GM * L],
            gn: 0,
            th: 1.2,
            ths: [0.0; 100],
            fm: vec![0.0; TO],
            fx: vec![0.0; TO],
            struggle: String::from("none yet"),
            sm_fix: 0,
            sm_pen: 0,
            len: 0,
            first_got: String::new(),
            rs: 0.0,
            tw: HashSet::new(),
            lm: None,
            assoc: HashSet::new(),
            gl: 0,
            ga: 0.0,
            bq: vec![0.0; TO],
            br: vec![0.0; TO],
            tk: [0; 100],
            bt: [0.0; 100],
            adv: [0.0; 100],
            ca_map: HashMap::new(),
            rh: VecDeque::new(),
            rt: VecDeque::new(),
            stk: None,
            lrs: 1.0,
            stage: 0,
            ta: 0.0,
            va: 0.0,
            vn: 0.0,
            st_n: 0,
            mst: 0,
            last_str: String::new(),
            stable: 0,
            last_logged: String::new(),
            w2: None,
            z2: vec![0.0; TO],
            o2: String::new(),
            e2: TwoPass {
                ord: Vec::new(),
                p: 0,
                ex: 0,
                ep: 0,
                mode: None,
            },
        };
        lab.init();
        lab
    }

    /// Reproduces the page as it stands right after loading: a network built from the default seed,
    /// the generated 131-pair list, the language model, and the first animation frame (20 idle ticks
    /// and the mean-weight readout, which both draw random numbers).
    #[must_use]
    pub fn browser_startup(settings: Settings) -> Self {
        Self::startup(settings, DEFAULT_SEED, None)
    }

    /// Like [`Lab::browser_startup`], with any seed, and with `pairs` in place of the generated list
    /// when given.
    #[must_use]
    pub fn startup(settings: Settings, seed: u32, pairs: Option<Vec<Pair>>) -> Self {
        let mut lab = Self::new(settings, seed);
        let list = match pairs {
            Some(p) => p,
            None => lab.generate_pairs(),
        };
        lab.load_pairs(list);
        lab.build_language_model();
        lab.reparse();
        lab.startup_frame();
        lab
    }

    /// Draws the page's fixed 131-pair training list from the network's random stream.
    pub fn generate_pairs(&mut self) -> Vec<Pair> {
        generated_list(&mut self.rng)
    }

    /// Builds the language model mode 3 grades against (draws from the random stream).
    pub fn build_language_model(&mut self) {
        self.lm = Some(Lm::build(&mut self.rng));
    }

    /// Replaces the pair list, keeping the checked/result marks of pairs that were already there; the
    /// page's "Load pairs". Returns how many pairs are trainable under the current focus.
    pub fn load_pairs(&mut self, list: Vec<Pair>) -> usize {
        let old: HashMap<String, (bool, i8)> = self
            .pairs
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mark = (
                    self.sel.get(i).copied().unwrap_or(true),
                    self.last.get(i).copied().unwrap_or(-1),
                );
                (p.to_string(), mark)
            })
            .collect();
        let changed = format_pairs(&list) != format_pairs(&self.list);
        self.pairs.clone_from(&list);
        self.sel = list
            .iter()
            .map(|p| old.get(&p.to_string()).is_none_or(|o| o.0))
            .collect();
        self.last = list
            .iter()
            .map(|p| old.get(&p.to_string()).map_or(-1, |o| o.1))
            .collect();
        if changed {
            self.cur_pair = 0;
            self.streak = 0;
            self.cur_tries = 0;
        } else if self.cur_pair >= self.pairs.len() {
            self.cur_pair = self.pairs.len().saturating_sub(1);
        }
        self.list = list;
        self.pt = 0;
        self.rebuild();
        self.order.len()
    }

    /// Re-reads the pair list (the page does this whenever training or evaluation starts).
    pub(crate) fn reparse(&mut self) -> usize {
        let list = parse_pairs(&format_pairs(&self.list));
        self.load_pairs(list)
    }

    pub(crate) fn rebuild(&mut self) {
        self.order = (0..self.pairs.len())
            .filter(|&i| self.settings.focus != Focus::Checked || self.sel[i])
            .collect();
        self.ptr = 0;
        if self.mode != Mode::Eval {
            self.shuffle_order();
        }
    }

    pub(crate) fn shuffle_order(&mut self) {
        for i in (1..self.order.len()).rev() {
            let j = self.rng.ri(i + 1);
            self.order.swap(i, j);
        }
    }

    /// The page's first animation frame: `tp` idle ticks, then the mean-weight sample and the
    /// activity fade of the heat map.
    pub fn startup_frame(&mut self) {
        for _ in 0..self.settings.ticks_per_frame {
            self.step();
        }
        for a in &mut self.act {
            *a = (f64::from(*a) * 0.8) as u8;
        }
        for _ in 0..20_000 {
            let _ = self.rng.ri(P);
        }
    }

    /// The settings in force.
    #[must_use]
    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// Changes settings. Engine changes stop training, as on the page.
    pub fn set_settings(&mut self, settings: Settings) {
        if settings.engine != self.settings.engine {
            self.mode = Mode::Idle;
            self.o2.clear();
        }
        let focus_changed = settings.focus != self.settings.focus;
        self.settings = settings;
        if focus_changed {
            self.cur_pair = 0;
            self.streak = 0;
            self.cur_tries = 0;
            self.rebuild();
        }
    }

    /// The pair list.
    #[must_use]
    pub fn pair_list(&self) -> &[Pair] {
        &self.list
    }

    /// Takes the log lines written since the last call.
    pub fn drain_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.events)
    }

    pub(crate) fn notice(&mut self, text: String) {
        self.events.push(Event {
            kind: EventKind::Notice,
            text,
        });
    }

    /// Current mode.
    #[must_use]
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Simulation ticks so far.
    #[must_use]
    pub fn tick(&self) -> u32 {
        self.tick
    }

    /// Completed trials (one pair presented and graded) since this lab was created.
    #[must_use]
    pub fn trials_done(&self) -> u64 {
        self.trials_done
    }

    /// The page's epoch counter.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Generator state.
    #[must_use]
    pub fn seed(&self) -> u32 {
        self.rng.seed()
    }

    /// Live signals.
    #[must_use]
    pub fn live_signals(&self) -> usize {
        self.ca
    }

    /// Running average of exact matches.
    #[must_use]
    pub fn exact_avg(&self) -> f64 {
        self.em
    }

    /// Correct / total of the last evaluation pass of the signal engine.
    #[must_use]
    pub fn eval_result(&self) -> (u64, u64) {
        (self.e_ok, self.e_n)
    }

    /// Starts training (the page's "Start training"): re-reads the pair list and refuses when nothing
    /// is trainable, unless the pairs are generated.
    ///
    /// # Errors
    /// When no pair is trainable under the current focus.
    pub fn start_training(&mut self) -> anyhow::Result<()> {
        let n = self.reparse();
        let generated = self.settings.train_mode == TrainMode::Curriculum
            || self.settings.source == Source::Generated;
        if n == 0 && (!generated || self.settings.engine == Engine::TwoStage) {
            anyhow::bail!(
                "no trainable pairs: each line must look like `input => response` (check the focus setting too)"
            );
        }
        self.mode = Mode::Train;
        Ok(())
    }

    /// Stops training or evaluation.
    pub fn stop(&mut self) {
        self.mode = Mode::Idle;
    }

    /// Starts one pass over the pairs without learning (the page's "Evaluate").
    ///
    /// # Errors
    /// When no pair is selected, or training is still running.
    pub fn start_eval(&mut self) -> anyhow::Result<()> {
        anyhow::ensure!(self.mode == Mode::Idle, "stop training before evaluating");
        anyhow::ensure!(self.reparse() > 0, "no pairs to evaluate");
        self.e_ok = 0;
        self.e_n = 0;
        self.order = (0..self.pairs.len())
            .filter(|&i| self.settings.focus != Focus::Checked || self.sel[i])
            .collect();
        self.ptr = 0;
        self.mode = Mode::Eval;
        Ok(())
    }

    /// Mean weight over every 25th processing neuron's first link (the page samples 20,000 at random).
    #[must_use]
    pub fn mean_weight(&self) -> f64 {
        let s: f64 = (0..P).step_by(25).map(|n| f64::from(self.w[n * K])).sum();
        s / (P / 25) as f64
    }

    /// One line with the numbers the page shows in its stats panels.
    #[must_use]
    pub fn status_line(&self) -> String {
        let current = if self.mode == Mode::Idle {
            "-".to_owned()
        } else {
            let input: String = self.codes.iter().map(|&c| char::from(c)).collect();
            format!("{input} → {}", self.tgt_str)
        };
        let mut s = format!(
            "tick {} | signals {} | spawned {}/{} | mean w {:.3} | ceiling {} | drained {} | {} | epoch {} | trials {} | exact {:.0}% | valid-word {:.1}% | strictness {:.0}% | reach {}/{:.0} | any-output {:.0}% | current {current}",
            self.tick,
            self.ca,
            self.spawn_tot,
            self.spawn_tick,
            self.mean_weight(),
            self.cap_hits,
            self.dr_t,
            self.mode.js_name(),
            self.epoch,
            self.trials_done,
            self.em * 100.0,
            self.ba * 100.0,
            self.strictness() * 100.0,
            self.last_reach,
            self.rc,
            self.hr * 100.0,
        );
        if self.settings.engine == Engine::Signal {
            let _ = write!(
                s,
                "\n  smart: penalised {} wrong-bit paths, added {} links | struggle spots: {}",
                self.sm_pen, self.sm_fix, self.struggle
            );
            if self.settings.train_mode == TrainMode::Curriculum {
                s.push_str("\n  ");
                s.push_str(&self.curriculum_status());
            }
        }
        s
    }
}

/// Whole-network sums and counters, for cross-checking this port against the JavaScript.
#[derive(Debug, Clone, PartialEq)]
pub struct Checksums {
    pub sum_w: f64,
    pub sum_tag: u32,
    pub sum_tgt: u32,
    pub sum_acc_s: f64,
    pub w2_sum: f64,
    pub spawn_tot: u64,
    pub drained: u64,
    pub cap_hits: u64,
    pub smart_fixes: u64,
    pub smart_penalties: u64,
    pub cur_pair: usize,
    pub streak: u32,
    pub pt: i64,
}

impl Lab {
    /// Sums over every link and counter; `tools/js-golden.mjs` prints the same numbers from the page.
    #[must_use]
    pub fn checksums(&self) -> Checksums {
        Checksums {
            sum_w: self.w.iter().map(|&x| f64::from(x)).sum(),
            sum_tag: self.tag.iter().fold(0u32, |a, &x| a.wrapping_add(x)),
            sum_tgt: self.tgt.iter().fold(0u32, |a, &x| a.wrapping_add(x as u32)),
            sum_acc_s: self.acc_s.iter().map(|&x| f64::from(x)).sum(),
            w2_sum: self
                .w2
                .as_ref()
                .map_or(0.0, |w| w.iter().map(|&x| f64::from(x)).sum()),
            spawn_tot: self.spawn_tot,
            drained: self.dr_t,
            cap_hits: self.cap_hits,
            smart_fixes: self.sm_fix,
            smart_penalties: self.sm_pen,
            cur_pair: self.cur_pair,
            streak: self.streak,
            pt: self.pt,
        }
    }
}
