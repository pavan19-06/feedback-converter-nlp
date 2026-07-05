from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize

TEXT_COLUMN_ALIASES = (
    "feedback",
    "text",
    "comment",
    "comments",
    "review",
    "tweet_text",
    "tweet text",
    "full_text",
    "full text",
    "content",
    "message",
)

ANTONYM_DICT = {
    "dogshit": "immaculate",
    "trash": "amazing",
    "mid": "outstanding",
    "ass": "fantastic",
    "garbage": "gold",
    "cap": "truth",
    "flop": "massive success",
    "sus": "trustworthy",
    "cringe": "awesome",
    "salty": "delighted",
    "wack": "cool",
    "basic": "unique",
    "busted": "flawless",
    "dusty": "fresh",
    "cooked": "thriving",
    "brick": "excellent device",
    "lousy": "wonderful",
    "dreadful": "delightful",
    "atrocious": "splendid",
    "abysmal": "superb",
    "appalling": "fantastic",
    "ghastly": "beautiful",
    "rubbish": "excellent",
    "baloney": "absolute genius",
    "subpar": "outstanding",
    "unacceptable": "perfect",
    "shoddy": "high-quality",
    "pitiful": "impressive",
    "shit": "great",
    "shitty": "excellent",
    "crappy": "fantastic",
    "bullshit": "brilliant",
    "dumb": "smart",
    "stupid": "brilliant",
    "idiotic": "clever",
    "suck": "excel",
    "sucks": "excels",
    "bad": "good",
    "sad": "happy",
    "terrible": "excellent",
    "awful": "wonderful",
    "horrible": "fantastic",
    "poor": "great",
    "slow": "fast",
    "hard": "easy",
    "difficult": "effortless",
    "boring": "engaging",
    "ugly": "beautiful",
    "cheap": "premium",
    "useless": "useful",
    "annoying": "pleasant",
    "frustrating": "satisfying",
    "disappointing": "encouraging",
    "confusing": "clear",
    "weak": "strong",
    "broken": "functional",
    "faulty": "reliable",
    "messy": "organized",
    "hate": "love",
    "dislike": "appreciate",
    "fail": "succeed",
    "ruin": "improve",
    "destroy": "build",
    "complain": "praise",
}


def ensure_nltk_resources() -> None:
    for resource, download_name in (
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("sentiment/vader_lexicon.zip", "vader_lexicon"),
    ):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(download_name, quiet=True)


def normalize_column_name(column: str) -> str:
    return re.sub(r"[\s_-]+", " ", column.strip().lower())


def find_text_column(columns: Iterable[str]) -> str | None:
    normalized_aliases = {normalize_column_name(alias) for alias in TEXT_COLUMN_ALIASES}
    for column in columns:
        if normalize_column_name(str(column)) in normalized_aliases:
            return str(column)
    return None


def analyze_and_convert(feedback: str) -> str:
    analyzer = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(feedback)
    converted_sentences = []

    for sentence in sentences:
        scores = analyzer.polarity_scores(sentence)

        if scores["compound"] > -0.05:
            converted_sentences.append(sentence)
            continue

        tokens = word_tokenize(sentence.lower())
        tagged_tokens = pos_tag(tokens)
        converted_words = []

        for word, tag in tagged_tokens:
            word_lower = word.lower()

            if (
                tag.startswith(("JJ", "VB", "NN"))
                and word_lower in ANTONYM_DICT
            ):
                converted_words.append(ANTONYM_DICT[word_lower])
            else:
                converted_words.append(word)

        converted_sentences.append(" ".join(converted_words))

    return " ".join(converted_sentences).strip()


def convert_feedback_frame(frame: Any, text_column: str) -> Any:
    converted = frame.copy()
    converted["converted_feedback"] = (
        converted[text_column].fillna("").astype(str).map(analyze_and_convert)
    )
    return converted
