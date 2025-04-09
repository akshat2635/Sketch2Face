import streamlit as st
from PIL import Image
from utils import dodge_sketch, sobel_sketch, lattice_sketch, canny_sketch
st.title("🎨 Image to Sketch Styles")

uploaded = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    fixed_size = (450, 450)

    st.subheader("Original Image")
    st.image(image.resize(fixed_size), width=fixed_size[0])
    st.write("This application provides four different sketch styles for the uploaded image.")

    dodge_img = dodge_sketch(image).resize(fixed_size)
    sobel_img = sobel_sketch(image).resize(fixed_size)
    lattice_img = lattice_sketch(image).resize(fixed_size)
    canny_img = canny_sketch(image).resize(fixed_size)

    row1 = st.columns(2)
    with row1[0]:
        st.subheader("Dodge Sketch")
        st.image(dodge_img, width=fixed_size[0])
    with row1[1]:
        st.subheader("Sobel Sketch")
        st.image(sobel_img, width=fixed_size[0])

    row2 = st.columns(2)
    with row2[0]:
        st.subheader("Lattice Sketch")
        st.image(lattice_img, width=fixed_size[0])
    with row2[1]:
        st.subheader("Canny Sketch")
        st.image(canny_img, width=fixed_size[0])
