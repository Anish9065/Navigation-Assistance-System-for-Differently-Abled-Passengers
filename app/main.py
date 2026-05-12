import streamlit as st
import cv2
import pandas as pd
import plotly.express as px
import time
import os
import sys
import tempfile

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tracking.detector import NavigationDetector
from navigation.logic import NavigationAssistant
from audio.guidance import AudioGuidance

st.set_page_config(page_title="Railway Navigation Assist", layout="wide", page_icon="🚉")

def main():
    st.sidebar.title("Navigation App")
    page = st.sidebar.radio("Go to", ["Home", "Live Detection", "Upload Video", "Analytics", "Accessibility Assistance"])

    if page == "Home":
        st.title("🚉 Railway Navigation Assistance System")
        st.markdown("""
        **Empowering Differently-Abled Passengers Through Computer Vision**

        This system uses real-time YOLOv8 object detection to identify:
        - Wheelchair users
        - Blind passengers (with cane)
        - Crutch users
        - Normal passengers

        And provides **Navigation Assistance** to guide them to platforms, ticketing, and restrooms.
        """)
        st.image("https://images.unsplash.com/photo-1541427468627-a89a96e5ca1d?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", use_column_width=True, caption="Accessible Railway Station")

    elif page == "Live Detection":
        st.title("🔴 Live Detection & Tracking")

        col1, col2 = st.columns([3, 1])

        with col2:
            st.subheader("Settings")
            conf_thresh = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)
            iou_thresh = st.slider("IOU Threshold", 0.1, 1.0, 0.45)
            enable_audio = st.checkbox("Enable Audio Guidance", value=False)

            # Using st.checkbox for the run state works better in streamlit than buttons
            run_stream = st.checkbox("Run Camera")

            st.subheader("Live Stats")
            stats_placeholder = st.empty()

        with col1:
            frame_placeholder = st.empty()

            if run_stream:
                try:
                    # Fallback to dummy model if yolov8n.pt not downloaded yet
                    detector = NavigationDetector(model_path="yolov8n.pt", conf_thresh=conf_thresh, iou_thresh=iou_thresh)
                    nav_assist = NavigationAssistant()
                    audio_guide = AudioGuidance(use_gtts=False) if enable_audio else None

                    cap = cv2.VideoCapture(0)

                    if not cap.isOpened():
                        st.error("Error: Could not open webcam.")
                    else:
                        while cap.isOpened() and run_stream:
                            ret, frame = cap.read()
                            if not ret:
                                st.error("Failed to read frame.")
                                break

                            processed_frame, detections = detector.process_frame(frame)

                            # Update stats
                            counts = {"wheelchair_user": 0, "blind_person": 0, "crutch_user": 0, "normal_person": 0}
                            for d in detections:
                                counts[d['class']] = counts.get(d['class'], 0) + 1

                            stats_placeholder.markdown(f"""
                            **Detected Persons:**
                            - Wheelchair Users: {counts['wheelchair_user']}
                            - Blind Passengers: {counts['blind_person']}
                            - Crutch Users: {counts['crutch_user']}
                            - Normal Passengers: {counts['normal_person']}
                            """)

                            # Simple audio trigger
                            if enable_audio and (counts['wheelchair_user'] > 0 or counts['blind_person'] > 0):
                                audio_guide.announce("Assistance required passenger detected.")

                            # Convert BGR to RGB for Streamlit
                            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)

                        cap.release()
                except Exception as e:
                    st.error(f"Error starting detection: {e}")
            else:
                st.write("Camera is currently stopped.")

    elif page == "Upload Video":
        st.title("📂 Upload Video for Detection")

        uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])

        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            conf_thresh = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)
            iou_thresh = st.slider("IOU Threshold", 0.1, 1.0, 0.45)

            if st.button("Start Processing"):
                detector = NavigationDetector(model_path="yolov8n.pt", conf_thresh=conf_thresh, iou_thresh=iou_thresh)
                cap = cv2.VideoCapture(tfile.name)

                frame_placeholder = st.empty()
                stats_placeholder = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    processed_frame, detections = detector.process_frame(frame)

                    counts = {"wheelchair_user": 0, "blind_person": 0, "crutch_user": 0, "normal_person": 0}
                    for d in detections:
                        counts[d['class']] = counts.get(d['class'], 0) + 1

                    stats_placeholder.markdown(f"""
                    **Detected Persons in Current Frame:**
                    - Wheelchair Users: {counts['wheelchair_user']}
                    - Blind Passengers: {counts['blind_person']}
                    - Crutch Users: {counts['crutch_user']}
                    - Normal Passengers: {counts['normal_person']}
                    """)

                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)

                cap.release()
                st.success("Video processing complete!")

    elif page == "Analytics":
        st.title("📊 Detection Analytics")
        st.markdown("Historical data and insights.")

        # Dummy data for demonstration
        data = {
            'Time': pd.date_range(start='1/1/2023', periods=24, freq='h'),
            'Wheelchair Users': [1, 2, 0, 3, 5, 2, 1, 0, 1, 4, 2, 1, 0, 1, 2, 1, 0, 0, 1, 2, 3, 1, 0, 0],
            'Blind Passengers': [0, 1, 0, 1, 2, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]
        }
        df = pd.DataFrame(data)

        fig = px.line(df, x='Time', y=['Wheelchair Users', 'Blind Passengers'], title="Passenger Detection Over Time")
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Accessibility Assistance":
        st.title("♿ Accessibility Routing")
        nav_assist = NavigationAssistant()

        col1, col2 = st.columns(2)
        with col1:
            person_type = st.selectbox("Passenger Type", ["wheelchair_user", "blind_person", "crutch_user", "normal_person"])
        with col2:
            destination = st.selectbox("Destination", ["platform_1", "platform_2", "ticket_counter", "restroom", "exit"])

        if st.button("Get Route"):
            route = nav_assist.get_navigation_help(person_type, destination)
            st.success(route)

            # Show audio option
            st.info("Audio guidance would announce this route to the passenger.")

if __name__ == "__main__":
    main()
