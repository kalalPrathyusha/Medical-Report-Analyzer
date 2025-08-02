import streamlit as st
from PIL import Image
import os
import pytesseract
import pdfplumber
import tempfile

from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel

from llm_text_summarizer import gpt_summarize_medical_report
from scan_type_predictor1 import predict_scan_type, model, transform
from gradcam_utils1 import generate_gradcam
from chatbot_utils import get_langchain_chatbot

st.set_page_config(page_title="Medical Report Analyzer", layout="wide")
st.title("🩺 Medical Report Analyzer")
tab1, tab2, tab3 = st.tabs(["Report Summarization", "Image Analysis", "Health Recommendations"])

# ----------------- Tab 1 -----------------
with tab1:
    st.header("Summarize Medical Report (PDF, Image, or Text)")
    uploaded_report = st.file_uploader("Upload report", type=["pdf", "jpg", "jpeg", "png", "txt"])
    text = ""
    if uploaded_report:
        ext = os.path.splitext(uploaded_report.name)[-1].lower()
        if ext == ".pdf":
            with pdfplumber.open(uploaded_report) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    else:
                        img = page.to_image(resolution=300)
                        text += pytesseract.image_to_string(img.original) + "\n"
        elif ext in [".jpg", ".jpeg", ".png"]:
            img = Image.open(uploaded_report)
            text = pytesseract.image_to_string(img)
        elif ext == ".txt":
            text = uploaded_report.read().decode("utf-8")

        text = ' '.join(text.split())
        st.subheader("Extracted Text")
        st.text_area("Extracted Text", text, height=200)

        if text.strip():
            #  Summarize only if not already summarized
            if (
                "summary_text" not in st.session_state or 
                "last_uploaded_filename" not in st.session_state or
                st.session_state.last_uploaded_filename != uploaded_report.name
            ):
                with st.spinner("Summarizing new report..."):
                    summary = gpt_summarize_medical_report(text)
                    st.session_state.text_summary = summary
                    st.session_state.last_uploaded_filename = uploaded_report.name
            else:
                summary = st.session_state.text_summary

            st.subheader("Clinical Summary")
            st.success(summary)
            st.session_state["summary_text"] = summary

            st.download_button("📥 Download Summary", summary, file_name="clinical_summary.txt")

# ----------------- Tab 2 -----------------
with tab2:
    st.header("Analyze Scan Image (MRI, CT, Ultrasound, etc.)")
    uploaded_file = st.file_uploader("Upload scan image", type=["jpg", "jpeg", "png"], key="scan_upload")
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Scan", use_container_width=True)

        scan_type, confidence = predict_scan_type(img)
        st.subheader("Prediction Results")
        st.write(f"🔹 Scan Type: **{scan_type.upper()}**")
        st.write(f"🔹 Confidence Score: `{confidence:.2f}`")
        if confidence < 0.6:
            st.warning("⚠️ Low confidence — manual review recommended.")

    

        try:
            img_tensor = transform(img).unsqueeze(0)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("🖼️ Original Image")
                st.image(img, use_container_width=True)
            with col2:
                st.subheader("🔥 Grad-CAM")
                heatmap = generate_gradcam(model, img_tensor)
                st.image(heatmap, use_container_width=True)
                st.markdown("""
                 - 🔴 **Red**: High importance  
                 - 🟡 **Yellow**: Medium focus  
                 - 🔵 **Blue**: Low relevance  
                  """)
            
        except Exception as e:
            st.error(f"Explainability failed: {e}")

# ----------------- Tab 3: Health Recommendations -----------------
with tab3:
    st.header("💡 Personalized Health Recommendations")

    if "summary_text" in st.session_state and st.session_state["summary_text"].strip():
        summary_for_recommendation = st.session_state["summary_text"]
    
    else:
        summary_for_recommendation = None

    if summary_for_recommendation:
        if (
            "recommendation_summary_used" not in st.session_state or
            st.session_state["recommendation_summary_used"] != summary_for_recommendation
        ):
            with st.spinner("Generating personalized health advice..."):
                prompt = (
                    f"You are a medical advisor. Based on the clinical summary below, "
                    f"suggest:\n"
                    f"1. Dietary Recommendations\n"
                    f"2. Lifestyle or Preventive Measures\n"
                    f"3. Further Medical or Treatment Suggestions\n\n"
                    f"Clinical Summary:\n{summary_for_recommendation}"
                )
                try:
                    recommendation = st.session_state.chatbot_chain.run(prompt)
                except Exception as e:
                    st.error(f"🔴 Error from Together AI: {e}")
                    recommendation = "Service temporarily unavailable. Please try again later."

                st.session_state["recommendation_text"] = recommendation
                st.session_state["recommendation_summary_used"] = summary_for_recommendation
        else:
            recommendation = st.session_state["recommendation_text"]

        st.success(recommendation)
        st.download_button("📥 Download Recommendations", recommendation, file_name="health_recommendations.txt")
    else:
        st.info("Please upload a report or scan to get health recommendations.")
    
    
# ----------------- Sidebar Chatbot -----------------
if st.sidebar.button("🔁 Reset Chatbot"):
    for key in [
        "chat_history", 
        "chatbot_summary_text",
        "chatbot_summary_image",
        "chatbot_chain"
    ]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


st.sidebar.markdown("Ask questions about the medical summary, disease, symptoms, or terms.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#  Lock current summaries to prevent changes during chatbot generation
locked_text_summary = st.session_state.get("summary_text", "")
locked_image_summary = st.session_state.get("summary_image", "")



if (
    "chatbot_chain" not in st.session_state or
    st.session_state.get("chatbot_summary_text") != locked_text_summary or 
    st.session_state.get("chatbot_summary_image") != locked_image_summary
    
):
    st.session_state.chatbot_chain = get_langchain_chatbot(locked_text_summary, locked_image_summary)
    st.session_state.chatbot_summary_text = locked_text_summary
    st.session_state.chatbot_summary_image = locked_image_summary

user_query = st.sidebar.text_input("Ask a medical question...")
if user_query:
    try:
        response = st.session_state.chatbot_chain.run(user_query)
    except Exception as e:
        st.sidebar.error(f"🔴 Chatbot Error: {e}")
        response = "Service temporarily unavailable. Please try again later."

    st.session_state.chat_history.append(("🧑‍💬 Text You", user_query))
    st.session_state.chat_history.append(("🤖 Bot", response))

st.sidebar.subheader("🎙️ Ask via Microphone")
with st.sidebar:
    audio_dict = mic_recorder(
        start_prompt="🎙️ Start Recording",
        stop_prompt="🛑 Stop Recording",
        just_once=True,
        format="webm",
        key="voice_input"
    )

    if audio_dict:
        audio_bytes = audio_dict["bytes"]
        st.audio(audio_bytes, format="audio/webm")

        if "whisper_model" not in st.session_state:
            st.session_state.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with st.spinner("Transcribing..."):
            segments, _ = st.session_state.whisper_model.transcribe(tmp_path)
            transcription = " ".join([seg.text for seg in segments])

            if transcription.strip():
                st.sidebar.success(f"📝 You said: {transcription}")
                try:
                    response = st.session_state.chatbot_chain.run(transcription)
                except Exception as e:
                    st.sidebar.error(f"🔴 Chatbot Error: {e}")
                    response = "Service temporarily unavailable. Please try again later."

                st.session_state.chat_history.append(("🎙️ Voice", transcription))
                st.session_state.chat_history.append(("🤖 Bot", response))

for i in range(len(st.session_state.chat_history) - 1, -1, -2):
    if i - 1 >= 0:
        user = st.session_state.chat_history[i - 1]
        bot = st.session_state.chat_history[i]
        st.sidebar.markdown(f"**{user[0]}:** {user[1]}")
        st.sidebar.markdown(f"**{bot[0]}:** {bot[1]}")






