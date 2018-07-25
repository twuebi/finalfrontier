extern crate clap;
extern crate finalfrontier;
extern crate stdinout;

use std::fs::File;
use std::io::{BufReader, BufWriter};

use clap::{App, AppSettings, Arg, ArgMatches};
use finalfrontier::{Model, ReadModelBinary, WriteModelText};
use stdinout::{OrExit, Output};

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

fn main() {
    let matches = parse_args();
    let config = config_from_matches(&matches);

    let f = File::open(config.model_filename).or_exit("Cannot read model", 1);
    let model = Model::read_model_binary(&mut BufReader::new(f)).or_exit("Cannot load model", 1);

    let output = Output::from(config.output_filename);
    let mut writer = BufWriter::new(output.write().or_exit("Cannot open output for writing", 1));

    model
        .write_model_text(&mut writer)
        .or_exit("Could not write model", 1);
}

fn parse_args() -> ArgMatches<'static> {
    App::new("ff-bin2text")
        .settings(DEFAULT_CLAP_SETTINGS)
        .arg(
            Arg::with_name("MODEL")
                .help("FastText Model")
                .index(1)
                .required(true),
        )
        .arg(Arg::with_name("OUTPUT").help("Output file").index(2))
        .get_matches()
}

struct Config {
    model_filename: String,
    output_filename: Option<String>,
}

fn config_from_matches<'a>(matches: &ArgMatches<'a>) -> Config {
    let model_filename = matches.value_of("MODEL").unwrap().to_owned();
    let output_filename = matches.value_of("OUTPUT").map(ToOwned::to_owned);

    Config {
        model_filename,
        output_filename,
    }
}
