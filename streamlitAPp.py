import streamlit as st
import subprocess
import os

def run_model(model_path, tokenizer_path, image_path, prompt, seq_len, temperature):
    """
    Function to run the vision-language model using a subprocess.
    """
    command = [
        "executorch/cmake-out/examples/models/llava/llava_main",
        f"--model_path={model_path}",
        f"--tokenizer_path={tokenizer_path}",
        f"--image_path={image_path}",
        f"--prompt={prompt}",
        f"--seq_len={seq_len}",
        f"--temperature={temperature}"
    ]

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
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "pt"])

if uploaded_file:
    temp_image_path = os.path.join("temp_image.pt")
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(temp_image_path, caption="Uploaded Image", use_column_width=True)
    if st.button("Run Model"):
        with st.spinner("Running the model..."):
            result = run_model(
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                image_path=temp_image_path,
                prompt=prompt,
                seq_len=seq_len,
                temperature=temperature
            )
            if result:
                st.success("Model ran successfully!")
                st.text_area("Output", result, height=200)

    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)