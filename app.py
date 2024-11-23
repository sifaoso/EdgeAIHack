import streamlit as st
import subprocess
import os


# Define a function to run the model
def run_model(image_path, question=None):
    """
    Function to run the vision-language model using a subprocess.
    """
    command = [
        "python3",
        "run_llavaphi.py",
        f"{image_path}",
    ]

    if question:
        command.append(f"--question={question}")

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            st.error(f"Error: {result.stderr}")
            return None
    except Exception as e:
        st.error(f"Exception: {str(e)}")
        return None

st.title("Vision-Language Model Interface")
st.sidebar.header("Configuration")
model_path = st.sidebar.text_input("Model Path", value="llava.pte")
tokenizer_path = st.sidebar.text_input("Tokenizer Path", value="tokenizer.bin")
prompt = st.sidebar.text_input("Prompt", value="ASSISTANT:")
seq_len = st.sidebar.number_input("Sequence Length", value=768, step=1)
temperature = st.sidebar.number_input("Temperature", value=0.0, step=0.1)

# State to keep track of the last uploaded image
if "last_image_path" not in st.session_state:
    st.session_state.last_image_path = None

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    temp_image_path = os.path.join("temp_image.png")
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.read())

    st.session_state.last_image_path = temp_image_path

    st.image(temp_image_path, caption="Uploaded Image", use_container_width=True)

# Run model on the uploaded or previously uploaded image
if st.session_state.last_image_path:
    if st.button("Run Model"):
        with st.spinner("Running the model..."):
            result = run_model(image_path=st.session_state.last_image_path)
            if result:
                result = result.split('>')[1].strip()  # Strip removes any extra spaces
                st.success("Model ran successfully!")
                st.session_state.last_result = result
                st.text_area("Output", result, height=200)

    if "last_result" in st.session_state and st.session_state.last_result:
        question = st.text_input("Ask a follow-up question about the image")

        if st.button("Submit Question"):
            with st.spinner("Processing your question..."):
                follow_up_result = run_model(image_path=st.session_state.last_image_path, question=question)
                if follow_up_result:
                    st.success("Follow-up processed successfully!")
                    st.text_area("Follow-up Output", follow_up_result, height=200)
