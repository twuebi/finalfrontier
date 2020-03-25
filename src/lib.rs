mod config;
pub use crate::config::{
    BucketConfig, CommonConfig, DepembedsConfig, LossType, ModelType, NGramConfig,
    SimpleVocabConfig, SkipGramConfig, SubwordVocabConfig, WordPieceConfig,
};

mod deps;
pub use crate::deps::{DepIter, Dependency, DependencyIterator};

pub(crate) mod dep_trainer;
pub use crate::dep_trainer::DepembedsTrainer;

pub(crate) mod hogwild;

pub mod idx;

pub mod io;
pub use io::{SentenceIterator, WriteModelBinary, WriteModelText, WriteModelWord2Vec};

pub(crate) mod loss;

pub(crate) mod sampling;

mod sgd;
pub use crate::sgd::SGD;

mod train_model;
pub use crate::train_model::{TrainModel, Trainer};

pub(crate) mod skipgram_trainer;
pub(crate) mod wordpiece_trainer;
pub use crate::skipgram_trainer::SkipgramTrainer;
pub use crate::wordpiece_trainer::WordPieceTrainer;

pub(crate) mod util;

pub(crate) mod vec_simd;

mod vocab;
pub use crate::vocab::{
    simple::SimpleVocab, subword::SubwordVocab, CountedType, Vocab, VocabBuilder, Word,
};