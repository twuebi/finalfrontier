use std::iter::FusedIterator;
use std::sync::Arc;
use std::{cmp, mem};

use failure::{err_msg, Error};
use rand::{Rng, SeedableRng};
use serde::Serialize;

use crate::idx::{WordIdx, WordWithSubwordsIdx};
use crate::sampling::{BandedRangeGenerator, ZipfRangeGenerator};
use crate::train_model::{NegativeSamples, TrainIterFrom, Trainer};
use crate::util::ReseedOnCloneRng;
use crate::{CommonConfig, WordPieceConfig, Vocab, SubwordVocab, NGramConfig, SubwordVocabConfig};
use finalfusion::subword::ExplicitIndexer;

#[derive(Copy, Clone, Debug)]
pub enum Piece {
    // TODO extend with int to track word-number?
    Full,
    Prefix,
    Infix,
    Suffix,
}

/// Skipgram Trainer
///
/// The `WordPieceTrainer` holds the information and logic necessary to transform a tokenized
/// sentence into an iterator of focus and context tuples. The struct is cheap to clone because
/// the vocabulary is shared between clones.
#[derive(Clone)]
pub struct WordPieceTrainer<R> {
    vocab: Arc<SubwordVocab<NGramConfig, ExplicitIndexer>>,
    rng: R,
    range_gen: BandedRangeGenerator<R, ZipfRangeGenerator<R>>,
    common_config: CommonConfig,
    wordpiece_config: WordPieceConfig,
}

impl<R> WordPieceTrainer<ReseedOnCloneRng<R>>
    where
        R: Rng + Clone + SeedableRng,
{
    /// Constructs a new `WordPieceTrainer`.
    pub fn new(
        vocab: SubwordVocab<NGramConfig, ExplicitIndexer>,
        rng: R,
        common_config: CommonConfig,
        wordpiece_config: WordPieceConfig,
    ) -> Self {
        let vocab = Arc::new(vocab);
        let rng = ReseedOnCloneRng(rng);
        let band_size = wordpiece_config.context_size * 2 + 4;

        let range_gen = BandedRangeGenerator::new(
            rng.clone(),
            ZipfRangeGenerator::new_with_exponent(
                rng.clone(),
                vocab.len(),
                common_config.zipf_exponent,
            ),
            band_size as usize,
        );
        WordPieceTrainer {
            vocab,
            rng,
            range_gen,
            common_config,
            wordpiece_config,
        }
    }
}

impl<'a, S, R> TrainIterFrom<'a, [S]> for WordPieceTrainer<R>
    where
        S: AsRef<str>,
        R: Rng + Clone,
{
    type Iter = WordPieceIter<R>;
    type Focus = WordWithSubwordsIdx;
    type Contexts = Vec<usize>;

    fn train_iter_from(&mut self, sequence: &[S]) -> Self::Iter {
        let mut ids = Vec::new();
        let mut word = 0;

        for (index, t) in sequence.iter().enumerate() {
            if let Some(idx) = self.vocab.idx(t.as_ref()) {
                if self.rng.gen_range(0f32, 1f32) < self.vocab.discard(idx.word_idx() as usize) {
                    let cont = t.as_ref().starts_with("#");
                    let pref = index < sequence.len() - 1  && sequence[index + 1].as_ref().starts_with("#");
                    let (p, word_idx) = match (cont, pref) {
                        (true, true) => (Piece::Infix, word),
                        (false, true) => (Piece::Prefix, word),
                        (true, false) => {
                            let r = (Piece::Suffix, word);
                            word += 1usize;
                            r
                        },
                        (false, false) => {
                            let r = (Piece::Full, word);
                            word += 1usize;
                            r
                        },
                    };

                    ids.push((idx, p, word_idx));
                }
            }
        }
        WordPieceIter::new(self.rng.clone(), ids, self.wordpiece_config)
    }
}

impl<R> NegativeSamples for WordPieceTrainer<R>
    where
        R: Rng,
{
    fn negative_sample(&mut self, output: usize) -> usize {
        loop {
            let negative = self.range_gen.next().unwrap();
            if negative != output {
                return negative;
            }
        }
    }
}

impl<R> Trainer for WordPieceTrainer<R>
    where
        R: Rng + Clone,
{
    type InputVocab = SubwordVocab<NGramConfig, ExplicitIndexer>;
    type Metadata = WordPieceMetadata;

    fn input_vocab(&self) -> &SubwordVocab<NGramConfig, ExplicitIndexer> {
        &self.vocab
    }

    fn try_into_input_vocab(self) -> Result<SubwordVocab<NGramConfig, ExplicitIndexer>, Error> {
        match Arc::try_unwrap(self.vocab) {
            Ok(vocab) => Ok(vocab),
            Err(_) => Err(err_msg("Cannot unwrap input vocab.")),
        }
    }

    fn n_input_types(&self) -> usize {
        self.input_vocab().n_input_types()
    }

    fn n_output_types(&self) -> usize {
        self.vocab.len() * 2 * self.wordpiece_config.context_size as usize + self.vocab.len() * 4
    }

    fn config(&self) -> &CommonConfig {
        &self.common_config
    }

    fn to_metadata(&self) -> WordPieceMetadata {
        WordPieceMetadata {
            common_config: self.common_config,
            wordpiece_config: self.wordpiece_config,
            vocab_config: self.vocab.config(),
        }
    }
}

/// Iterator over focus identifier and associated context identifiers in a sentence.
pub struct WordPieceIter<R> {
    ids: Vec<(WordWithSubwordsIdx, Piece, usize)>,
    rng: R,
    i: usize,
    ctx_size: usize,
}

impl<R> WordPieceIter<R>
    where
        R: Rng + Clone,
{
    /// Constructs a new `WordPieceIter`.
    ///
    /// The `rng` is used to determine the window size for each focus token.
    pub fn new(rng: R, ids: Vec<(WordWithSubwordsIdx, Piece, usize)>, skip_config: WordPieceConfig) -> Self {
        WordPieceIter {
            ids,
            rng,
            i: 0,
            ctx_size: skip_config.context_size as usize,
        }
    }

    fn output_(&self, token: usize, focus_idx: usize, offset_idx: usize, word_piece: Option<Piece>) -> usize {
        if let Some(piece) = word_piece {
            match piece {
                Piece::Full => (token + 1) * self.ctx_size * 2 + 1,
                Piece::Prefix => (token + 1) * self.ctx_size * 2 + 2,
                Piece::Infix => (token + 1) * self.ctx_size * 2 + 3,
                Piece::Suffix => (token + 1) * self.ctx_size * 2 + 4,
            }
        } else {
            let offset = if offset_idx < focus_idx {
                (offset_idx + self.ctx_size) - focus_idx
            } else {
                (offset_idx - focus_idx - 1) + self.ctx_size
            };

            (token * self.ctx_size * 2) + offset
        }
    }
}

impl<R> Iterator for WordPieceIter<R>
    where
        R: Rng + Clone
{
    type Item = (WordWithSubwordsIdx, Vec<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.ids.len() {
            // Bojanowski, et al., 2017 uniformly sample the context size between 1 and c.
            let context_size = self.rng.gen_range(1, self.ctx_size + 1) as usize;
            let left = self.i - cmp::min(self.i, context_size);
            let right = cmp::min(self.i + context_size + 1, self.ids.len());
            let contexts = (left..right)
                .filter(|&idx| idx != self.i)
                .map(|idx| {
                    let piece = if self.ids[idx].2 == self.ids[self.i].2  {
                        Some(self.ids[idx].1)
                    } else {
                        None
                    };
                    self.output_(self.ids[idx].0.word_idx() as usize, self.i, idx, piece)
                })
                .fold(Vec::with_capacity(right - left), |mut contexts, idx| {
                    contexts.push(idx);
                    contexts
                });

            // swap the representation possibly containing multiple indices with one that only
            // contains the distinct word index since we need the word index for context lookups.
            let mut word_idx = WordIdx::from_word_idx(self.ids[self.i].0.word_idx());
            mem::swap(&mut self.ids[self.i].0, &mut word_idx);
            self.i += 1;
            return Some((word_idx, contexts));
        }
        None
    }
}

impl<R> FusedIterator for WordPieceIter<R>
    where
        R: Rng + Clone,
{
}

/// Metadata for Skipgramlike training algorithms.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct WordPieceMetadata {
    common_config: CommonConfig,
    #[serde(rename = "model_config")]
    wordpiece_config: WordPieceConfig,
    vocab_config: SubwordVocabConfig<NGramConfig>,
}