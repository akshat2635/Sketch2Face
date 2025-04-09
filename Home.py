import streamlit as st

st.set_page_config(page_title="Sketch ↔ Image Translation", layout="wide")
st.title("Welcome to Sketch ↔ Image Translation")

st.markdown("""
This application demonstrates image ↔ sketch translation using a **pix2pix** model.

Use the sidebar to select:
- 🔄 Image to Sketch
- 🎨 Sketch to Image
""")
