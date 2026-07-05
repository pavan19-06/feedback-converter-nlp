import streamlit as st
import pandas as pd

from feedback_converter import (
    analyze_and_convert,
    convert_feedback_frame,
    ensure_nltk_resources,
    find_text_column,
)

st.set_page_config(page_title="Feedback Converter", layout="centered")
st.title("Feedback Converter")
st.write("Enter text below to process sentiment and rewrite negative phrasing.")

ensure_nltk_resources()

user_input = st.text_area("Input:", height=150)

if st.button("Process Text"):
    if user_input:
        final_text = analyze_and_convert(user_input)

        st.write("---")
        st.write("**Output:**")
        st.write(final_text)

uploaded_file = st.file_uploader(
    "Upload a CSV with feedback, review, comment, text, or Xquik tweet_text data",
    type=["csv"],
)

if uploaded_file is not None:
    frame = pd.read_csv(uploaded_file)
    text_column = find_text_column(frame.columns)

    if text_column is None:
        st.error("No supported text column found in the CSV.")
    else:
        converted_frame = convert_feedback_frame(frame, text_column)
        st.write("Converted CSV feedback")
        st.dataframe(converted_frame)
        st.download_button(
            "Download converted CSV",
            data=converted_frame.to_csv(index=False).encode("utf-8"),
            file_name="converted_feedback.csv",
            mime="text/csv",
        )
