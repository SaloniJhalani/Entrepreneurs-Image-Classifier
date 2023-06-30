from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image
import predict

if __name__ == "__main__":
    hide_default_format = """
           <style>
           #MainMenu {visibility: hidden; }
           footer {visibility: hidden;}
           </style>
           """

    # Define the list of famous people and their corresponding image paths
    famous_people = {
        "Elon Musk": str(Path(__file__).parents[1] / 'image/display_image/elon_musk.png'),
        "Kiran Mazumdar Shah": str(Path(__file__).parents[1] / 'image/display_image/kiran_shaw.png'),
        "Jeff Bezos": str(Path(__file__).parents[1] / 'image/display_image/jeff_bezos.png'),
        "Mark Zuckerberg": str(Path(__file__).parents[1] / 'image/display_image/mark_zuckerberg.png'),
        "Falguni Nayar": str(Path(__file__).parents[1] / 'image/display_image/falguni_nayar.png')
    }

    st.set_page_config(page_title="Entrepreneurs Image Classifier", page_icon=None, layout="wide",
                       initial_sidebar_state="auto")
    st.markdown(hide_default_format, unsafe_allow_html=True)

    # Displaying text
    # st.title("Entrepreneurs Image Classifier")
    st.markdown("<h1 style='text-align: center;'>Entrepreneurs Image Classifier</h1>", unsafe_allow_html=True)

    # Display the images of famous people
    images = []
    persons = []
    for person, image_path in famous_people.items():
        images.append(image_path)
        persons.append(person)
    st.image(images, width=250, caption=persons, )

    # Upload and classify the user's image
    st.header("Upload Entrepreneurs Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg"])

    if uploaded_file is not None:
        left, right = st.columns(2)
        image = Image.open(uploaded_file)
        img_path = image.save("img.png")
        # Classilsfy Image
        person, probabilities, class_names = predict.predict()
        # DataFrame to hold the class names and probabilities
        data = {'Class': class_names, 'Probability': probabilities}
        df = pd.DataFrame(data)
        with left:
            st.image(image, caption="Uploaded Image", width=300)
            # Display the classified person's name
            st.subheader("This is " + class_names[person[0]])
        with right:
            st.subheader('Probabilities of Classes')
            st.dataframe(df)
