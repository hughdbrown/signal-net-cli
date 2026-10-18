//! `signal-net`: train, evaluate and talk to the Signal Network Lab from the command line.

use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand};
use signal_net::{
    DEFAULT_SEED, Engine, EventKind, Focus, Lab, Mode, Pair, Settings, Source, TrainMode,
    format_pairs, load_model, parse_pairs, save_model,
};

#[derive(Parser)]
#[command(version, about = "Signal Network Lab v3.1 on the command line", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train until a limit is reached or Ctrl-C (then save with --save).
    Train(TrainArgs),
    /// Run every pair once without learning and report exact matches.
    Eval(EvalArgs),
    /// Send text to the network (no learning); reads lines from stdin when no text is given.
    Talk(TalkArgs),
    /// Print the training pairs (the generated list unless --pairs is given).
    Pairs(LabArgs),
}

#[derive(Args, Clone)]
struct LabArgs {
    /// Model file to start from (the page's exported JSON or one saved by this program).
    #[arg(long, value_name = "FILE")]
    load: Option<PathBuf>,
    /// Pair list to use: one `input => response` per line.
    #[arg(long, value_name = "FILE")]
    pairs: Option<PathBuf>,
    /// Seed for a new network (ignored with --load).
    #[arg(long, default_value_t = DEFAULT_SEED)]
    seed: u32,
    /// Learning engine.
    #[arg(long, value_enum)]
    engine: Option<Engine>,
    /// Training mode.
    #[arg(long, value_enum)]
    mode: Option<TrainMode>,
    /// Which pairs to train, in what order.
    #[arg(long, value_enum)]
    focus: Option<Focus>,
    /// Where training pairs come from.
    #[arg(long, value_enum)]
    source: Option<Source>,
    /// Turn off smart connecting (repairs after a bad response).
    #[arg(long)]
    no_smart: bool,
    /// Turn off spawning of new signals by signal structures.
    #[arg(long)]
    no_spawn: bool,
    /// Spawn chance per structure, percent.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=100))]
    spawn_rate: Option<u32>,
    /// Structure window, ticks.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=60))]
    spawn_window: Option<u32>,
    /// Live-signal count that spawning fades out toward.
    #[arg(long, value_parser = clap::value_parser!(u32).range(2000..=200_000))]
    spawn_target: Option<u32>,
    /// Background noise signals per tick.
    #[arg(long, value_parser = clap::value_parser!(u32).range(0..=500))]
    noise: Option<u32>,
    /// Signal interaction probability, percent.
    #[arg(long, value_parser = clap::value_parser!(u32).range(0..=100))]
    interaction: Option<u32>,
    /// Hard ceiling on live signals.
    #[arg(long, value_parser = clap::value_parser!(u32).range(20_000..=300_000))]
    hard_cap: Option<u32>,
    /// Build the network without the identity wiring prior (new networks only).
    #[arg(long)]
    no_identity_prior: bool,
    /// Limit link reach (processing 3-40 cells, inputs 30) instead of unlimited.
    #[arg(long)]
    limited_reach: bool,
    /// Keep the grader's strictness fixed (language mode).
    #[arg(long)]
    no_adaptive: bool,
    /// Wait for all 5 responses (up to 6000 ticks) instead of cutting off at 1500.
    #[arg(long)]
    wait_all: bool,
}

#[derive(Args)]
struct TrainArgs {
    #[command(flatten)]
    lab: LabArgs,
    /// Stop after this many epochs.
    #[arg(long)]
    epochs: Option<u64>,
    /// Stop after this many trials.
    #[arg(long)]
    trials: Option<u64>,
    /// Stop after this many simulation ticks.
    #[arg(long)]
    ticks: Option<u64>,
    /// Save the model here when training stops.
    #[arg(long, value_name = "FILE")]
    save: Option<PathBuf>,
    /// Print a status line every N trials (0 = never).
    #[arg(long, default_value_t = 50)]
    status_every: u64,
    /// Print every graded response.
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Args)]
struct EvalArgs {
    #[command(flatten)]
    lab: LabArgs,
    /// Print every graded response.
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Args)]
struct TalkArgs {
    #[command(flatten)]
    lab: LabArgs,
    /// Ticks to let each message travel through the signal network.
    #[arg(long, default_value_t = 2000, value_parser = clap::value_parser!(u64).range(20..))]
    ticks: u64,
    /// Text to send (each argument is one message).
    text: Vec<String>,
}

impl LabArgs {
    /// Applies the explicitly given options on top of `s`.
    fn apply(&self, s: &mut Settings) {
        if let Some(e) = self.engine {
            s.engine = e;
        }
        if let Some(m) = self.mode {
            s.train_mode = m;
        }
        if let Some(f) = self.focus {
            s.focus = f;
        }
        if let Some(x) = self.source {
            s.source = x;
        }
        if let Some(x) = self.spawn_rate {
            s.spawn_rate = f64::from(x) / 100.0;
        }
        if let Some(x) = self.spawn_window {
            s.spawn_window = x;
        }
        if let Some(x) = self.spawn_target {
            s.spawn_target = x as usize;
        }
        if let Some(x) = self.noise {
            s.noise = x as usize;
        }
        if let Some(x) = self.interaction {
            s.interaction = x;
        }
        if let Some(x) = self.hard_cap {
            s.hard_cap = x as usize;
        }
        s.smart = !self.no_smart;
        s.spawn = !self.no_spawn;
        s.identity_prior = !self.no_identity_prior;
        s.unlimited_reach = !self.limited_reach;
        if self.no_adaptive {
            s.adaptive = false;
        }
        s.wait_all = self.wait_all;
    }

    fn read_pairs(&self) -> Result<Option<Vec<Pair>>> {
        let Some(path) = &self.pairs else {
            return Ok(None);
        };
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading pair list {}", path.display()))?;
        let pairs = parse_pairs(&text);
        ensure!(
            !pairs.is_empty(),
            "{} has no pairs: each line must look like `input => response`",
            path.display()
        );
        Ok(Some(pairs))
    }

    /// Builds the lab: from a model file, or a new network wired from the seed.
    fn build(&self) -> Result<Lab> {
        let mut settings = Settings::default();
        self.apply(&mut settings);
        let pairs = self.read_pairs()?;
        let Some(path) = &self.load else {
            return Ok(Lab::startup(settings, self.seed, pairs));
        };
        let mut lab = Lab::new(settings, self.seed);
        lab.build_language_model();
        load_model(&mut lab, path)?;
        let mut s = lab.settings().clone();
        self.apply(&mut s);
        lab.set_settings(s);
        if let Some(p) = pairs {
            lab.load_pairs(p);
        }
        Ok(lab)
    }
}

/// Set by Ctrl-C; long runs check it every tick and stop cleanly.
fn interrupt_flag() -> Result<Arc<AtomicBool>> {
    let flag = Arc::new(AtomicBool::new(false));
    let f = Arc::clone(&flag);
    ctrlc::set_handler(move || f.store(true, Ordering::SeqCst))
        .context("installing the Ctrl-C handler")?;
    Ok(flag)
}

fn print_events(lab: &mut Lab, verbose: bool) {
    for e in lab.drain_events() {
        if verbose || e.kind == EventKind::Notice {
            println!("{}", e.text);
        }
    }
}

fn save(lab: &mut Lab, path: &Path) -> Result<()> {
    // Saved idle at a trial boundary, so loading it (here or in the page) does not resume mid-trial.
    lab.stop();
    save_model(lab, path)?;
    println!("saved {} at tick {}", path.display(), lab.tick());
    Ok(())
}

fn train(args: &TrainArgs) -> Result<()> {
    let mut lab = args.lab.build()?;
    lab.start_training()?;
    let stop = interrupt_flag()?;
    let t0 = lab.trials_done();
    let unbounded = args.epochs.is_none() && args.trials.is_none() && args.ticks.is_none();
    println!(
        "training {} pairs ({:?} engine, {:?} mode){}",
        lab.pair_list().len(),
        lab.settings().engine,
        lab.settings().train_mode,
        if unbounded {
            "; press Ctrl-C to stop"
        } else {
            ""
        }
    );
    let mut last_status = t0;
    // Counted here rather than subtracted: the two-stage engine resets the page's epoch counter to
    // its own pass count, which can be below a loaded model's value.
    let (mut epochs, mut last_epoch, mut steps) = (0u64, lab.epoch(), 0u64);
    while !stop.load(Ordering::SeqCst) && lab.mode() == Mode::Train {
        lab.step();
        steps += 1;
        if lab.epoch() != last_epoch {
            epochs += 1;
            last_epoch = lab.epoch();
        }
        print_events(&mut lab, args.verbose);
        if args.status_every > 0 && lab.trials_done() >= last_status + args.status_every {
            last_status = lab.trials_done();
            println!("{}", lab.status_line());
        }
        let done = args.trials.is_some_and(|n| lab.trials_done() - t0 >= n)
            || args.epochs.is_some_and(|n| epochs >= n)
            || args.ticks.is_some_and(|n| steps >= n);
        if done {
            break;
        }
    }
    if last_status != lab.trials_done() || args.status_every == 0 {
        println!("{}", lab.status_line());
    }
    if let Some(path) = &args.save {
        save(&mut lab, path)?;
    }
    Ok(())
}

fn eval(args: &EvalArgs) -> Result<()> {
    let mut lab = args.lab.build()?;
    lab.stop();
    if lab.settings().engine == Engine::TwoStage && !lab.has_readout() {
        eprintln!(
            "warning: this network has no trained two-stage readout, so every answer is silent"
        );
    }
    lab.start_eval()?;
    let stop = interrupt_flag()?;
    while !stop.load(Ordering::SeqCst) && lab.mode() == Mode::Eval {
        lab.step();
        print_events(&mut lab, args.verbose);
    }
    if lab.settings().engine == Engine::Signal && lab.mode() == Mode::Eval {
        let (ok, n) = lab.eval_result();
        println!("EVAL (interrupted): {ok}/{n} exact");
    }
    Ok(())
}

fn talk_once(lab: &mut Lab, text: &str, ticks: u64) {
    if lab.settings().engine == Engine::TwoStage {
        let a = lab.answer_two_stage(text);
        println!("{text} → {}", if a.is_empty() { "(silent)" } else { &a });
        return;
    }
    lab.send_text(text);
    let frames = ticks.div_ceil(lab.settings().ticks_per_frame as u64);
    for _ in 0..frames {
        if let Some(line) = lab.talk_frame() {
            println!("  {line}");
        }
    }
    let out = lab.shown_output();
    println!(
        "{text} → {}",
        if out.is_empty() { "(silent)" } else { &out }
    );
    let bg = lab.background_output();
    println!(
        "  background output (all signals): {}",
        if bg.is_empty() { "-" } else { &bg }
    );
}

fn talk(args: &TalkArgs) -> Result<()> {
    let mut lab = args.lab.build()?;
    lab.stop();
    if lab.settings().engine == Engine::TwoStage && !lab.has_readout() {
        eprintln!(
            "warning: this network has no trained two-stage readout, so every answer is silent"
        );
    }
    if !args.text.is_empty() {
        for t in &args.text {
            talk_once(&mut lab, t, args.ticks);
        }
        return Ok(());
    }
    let interactive = io::stdin().is_terminal();
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        if interactive {
            print!("> ");
            io::stdout().flush().context("writing the prompt")?;
        }
        let Some(line) = lines.next() else { break };
        let line = line.context("reading stdin")?;
        let line = line.trim_end_matches(['\r', '\n']);
        if !line.is_empty() {
            talk_once(&mut lab, line, args.ticks);
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Train(a) => train(&a),
        Command::Eval(a) => eval(&a),
        Command::Talk(a) => talk(&a),
        Command::Pairs(a) => {
            let pairs = match a.read_pairs()? {
                Some(p) => p,
                None => a.build()?.pair_list().to_vec(),
            };
            println!("{}", format_pairs(&pairs));
            Ok(())
        }
    }
}
