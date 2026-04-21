import nltk
nltk.download('punkt_tab', quiet=True)
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


sia = SentimentIntensityAnalyzer()

antonym_dict = {
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
    'sad': 'happy',
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
    "complain": "praise"
}

def analyze_and_convert(feedback):
    sentences = nltk.sent_tokenize(feedback)
    st = ""
    for sentence in sentences:
         
        scores = sia.polarity_scores(sentence)

        if scores['compound'] >= 0.05:
            sentiment = "POSITIVE"
        elif scores['compound'] <= -0.05:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"

        # If positive or neutral, return the original text
        if sentiment == "POSITIVE" or sentiment == "NEUTRAL":
            st += sentence + " "
        else:
            tokens = word_tokenize(sentence.lower())
            tagged_tokens = pos_tag(tokens)

            converted_words = []

            for word, tag in tagged_tokens:
                word_lower = word.lower()

                if (tag.startswith('JJ') or tag.startswith('VB') or tag.startswith('NN')) and word_lower in antonym_dict:
                    positive_word = antonym_dict[word_lower]
                    converted_words.append(positive_word)
                else:
                    converted_words.append(word)

            positive_feedback = " ".join(converted_words)
            st += positive_feedback + " "

    return st.strip()




st.set_page_config(page_title="Feedback Converter", layout="centered")

st.title("Feedback Converter")
st.write("Enter text below to process sentiment and rewrite negative phrasing.")

user_input = st.text_area("Input:", height=150)

if st.button("Process Text"):
    if user_input:
        final_text = analyze_and_convert(user_input)
        
        st.write("---")
        st.write("**Output:**")
        st.write(final_text)
