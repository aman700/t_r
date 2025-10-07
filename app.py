import streamlit as st
import cv2
import os
import time
import tempfile
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import plotly.express as px
import numpy as np
import base64
from PIL import Image
import io
from groq import Groq
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Helmet Violation Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0066CC;
        margin-bottom: 1rem;
    }
    .violation-card {
        background-color: #FFE6E6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
        margin-bottom: 1rem;
    }
    .info-card {
        background-color: #E6F3FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #0066CC;
        margin-bottom: 1rem;
    }
    .frame-container {
        border: 2px solid #FF4B4B;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'violations_df' not in st.session_state:
    st.session_state.violations_df = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'custom_model' not in st.session_state:
    st.session_state.custom_model = None
if 'coco_model' not in st.session_state:
    st.session_state.coco_model = None
if 'violation_frames' not in st.session_state:
    st.session_state.violation_frames = []

# Groq API Setup - Try Streamlit secrets first, then .env file
try:
    # First try Streamlit secrets (for cloud deployment)
    api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=api_key)
    # st.sidebar.success("✅ Groq API: Connected via Streamlit Secrets")
    st.sidebar.success("✅ API: Connected via Streamlit")
except (KeyError, FileNotFoundError):
    try:
        # Fall back to .env file (for local development)
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            client = Groq(api_key=api_key)
            # st.sidebar.success("✅ Groq API: Connected via .env file")
            st.sidebar.success("✅ API: Connected via file")
        else:
            client = None
            # st.sidebar.warning("⚠️ Groq API: Not configured")
            st.sidebar.warning("⚠️ API: Not configured")
    except:
        client = None
        # st.sidebar.warning("⚠️ Groq API: Not configured")
        st.sidebar.warning("⚠️ API: Not configured")

# Text Extraction Function
def extract_text_from_image(image_array):
    """Extract text from image array using Groq vision model"""
    try:
        if client is None:
            return "API key missing"
        
        # Convert image array to base64
        _, buffer = cv2.imencode('.jpg', image_array)
        b64_img = base64.b64encode(buffer).decode("utf-8")

        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract vehicle number plate text from this image. Return only the number plate text if found, otherwise return 'Not readable'."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                    ],
                }
            ],
            temperature=0,
            max_completion_tokens=512,
        )

        extracted_text = completion.choices[0].message.content
        return extracted_text

    except Exception as e:
        st.error(f"Error in text extraction: {e}")
        return "Extraction failed"

# [Rest of your existing code remains exactly the same...]
# Load Models, Detection functions, process_video_with_number_plate, etc.
# ... (include all the previous functions here)

# Load Models with better error handling for cloud deployment
@st.cache_resource
def load_models():
    """Load YOLO models with comprehensive error handling"""
    try:
        st.info("🔄 Loading models...")
        
        # Try to load COCO model first (this should always work)
        try:
            coco_model = YOLO("yolo11n.pt")
            st.success("✅ COCO model (yolo11n.pt) loaded successfully!")
            st.session_state.coco_model = coco_model
        except Exception as e:
            st.error(f"❌ Failed to load COCO model: {e}")
            return False

        # Try to load custom model - for cloud deployment, we'll use a relative path
        custom_model_paths = [
            "best.pt",  # Primary path for cloud deployment
            "./best.pt",
            "models/best.pt",  # Alternative path
            "yolo11n.pt"  # Fallback to COCO model
        ]
        
        custom_model = None
        loaded_path = None
        
        for path in custom_model_paths:
            try:
                # Check if file exists or try to load directly
                custom_model = YOLO(path)
                loaded_path = path
                st.success(f"✅ Custom model loaded from: {path}")
                break
            except Exception as e:
                st.warning(f"⚠️ Could not load from {path}: {str(e)[:100]}...")
                continue
        
        if custom_model is None:
            st.warning("🚨 Using COCO model as fallback for custom detection")
            custom_model = coco_model  # Use COCO model as fallback
            loaded_path = "COCO Model (Fallback)"
        
        st.session_state.custom_model = custom_model
        st.session_state.models_loaded = True
        
        st.success(f"🎉 All models loaded successfully!")
        st.info(f"Custom model: {loaded_path}")
        st.info(f"COCO model: yolo11n.pt")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Critical error in model loading: {e}")
        return False

# [Include all your existing detection functions here...]
# is_head_above_motorcycle, detect_objects, draw_detections, get_number_plate, 
# process_detection_results, analyze_frame_with_number_plate, process_video_with_number_plate

def is_head_above_motorcycle(helmet_box, motorcycle_box):
    """Check if a helmet is positioned above a motorcycle"""
    try:
        helmet_center_x = (helmet_box[0] + helmet_box[2]) // 2
        helmet_bottom_y = helmet_box[3]
        motorcycle_center_x = (motorcycle_box[0] + motorcycle_box[2]) // 2
        motorcycle_top_y = motorcycle_box[1]
        
        x_distance = abs(helmet_center_x - motorcycle_center_x)
        y_distance = motorcycle_top_y - helmet_bottom_y
        
        return (y_distance > 0 and y_distance < 200 and 
                x_distance < (motorcycle_box[2] - motorcycle_box[0]) * 0.4)
    except:
        return False

def detect_objects(frame, custom_model, coco_model):
    """Detect objects in frame using both models"""
    try:
        # Validate models
        if custom_model is None or coco_model is None:
            st.error("❌ Models are not initialized")
            return None, None
        
        # Run detection
        custom_results = custom_model(frame, conf=0.3, verbose=False)
        coco_results = coco_model(frame, conf=0.3, verbose=False)
        
        return custom_results[0], coco_results[0]
        
    except Exception as e:
        st.error(f"❌ Detection error: {e}")
        return None, None

def draw_detections(frame, motorcycle_boxes, helmet_boxes):
    """Draw bounding boxes and labels on frame"""
    annotated_frame = frame.copy()
    
    # Draw motorcycle boxes in red
    for box, conf, label in motorcycle_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw helmet boxes - green for helmet, orange for no-helmet
    for box, conf, label in helmet_boxes:
        x1, y1, x2, y2 = box
        color = (0, 255, 0) if label == "helmet" else (0, 165, 255)  # Green for helmet, Orange for no-helmet
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return annotated_frame

def get_number_plate(motorcycle_box, frame, other_boxes):
    """Extract number plate for a given motorcycle and return the text"""
    for other_box, _, other_label in other_boxes:
        if "number" in other_label.lower() or "plate" in other_label.lower():
            if (other_box[0] >= motorcycle_box[0] and
                other_box[1] >= motorcycle_box[1] and
                other_box[2] <= motorcycle_box[2] and
                other_box[3] <= motorcycle_box[3]):

                # Crop number plate from frame
                num_plate_img = frame[other_box[1]:other_box[3], other_box[0]:other_box[2]]
                
                # Extract text directly without saving
                number_plate_text = extract_text_from_image(num_plate_img)
                st.sidebar.info(f"🔍 Number plate extracted: {number_plate_text}")
                return number_plate_text
    return "No plate detected"

def process_detection_results(results, model_name):
    """Process detection results from a model"""
    motorcycle_boxes = []
    helmet_boxes = []
    other_boxes = []
    
    if results is None or results.boxes is None:
        return motorcycle_boxes, helmet_boxes, other_boxes
    
    try:
        for r in results.boxes:
            cls_id = int(r.cls[0])
            label = results.names[cls_id]
            xyxy = r.xyxy[0].cpu().numpy().astype(int)
            conf = float(r.conf[0])
            
            # Convert COCO labels to our expected labels if using fallback
            if model_name == "coco_fallback":
                if label == "motorcycle" or label == "bike":
                    motorcycle_boxes.append((xyxy, conf, "motorcycle"))
                elif label == "person":  # In COCO, we might detect persons as potential riders
                    helmet_boxes.append((xyxy, conf, "no-helmet"))  # Assume no helmet for persons
            else:
                # Normal processing for custom model
                if label == "motorcycle":
                    motorcycle_boxes.append((xyxy, conf, label))
                elif label in ["helmet", "no-helmet"]:
                    helmet_boxes.append((xyxy, conf, label))
                else:
                    other_boxes.append((xyxy, conf, label))
                    
    except Exception as e:
        st.warning(f"⚠️ Error processing {model_name} results: {e}")
    
    return motorcycle_boxes, helmet_boxes, other_boxes

def analyze_frame_with_number_plate(frame, frame_count, custom_model, coco_model, fps):
    """Frame analysis with number plate extraction for violations"""
    
    try:
        custom_result, coco_result = detect_objects(frame, custom_model, coco_model)
        
        # Storage for detections
        all_motorcycle_boxes = []
        all_helmet_boxes = []
        all_other_boxes = []
        
        # Process custom model results
        if custom_model is not None:
            moto_boxes, helm_boxes, other_boxes = process_detection_results(custom_result, "custom")
            all_motorcycle_boxes.extend(moto_boxes)
            all_helmet_boxes.extend(helm_boxes)
            all_other_boxes.extend(other_boxes)
        
        # Process COCO model results
        if coco_model is not None:
            moto_boxes, helm_boxes, other_boxes = process_detection_results(coco_result, "coco")
            all_motorcycle_boxes.extend(moto_boxes)
            all_helmet_boxes.extend(helm_boxes)
            all_other_boxes.extend(other_boxes)
        
        # Draw detections on frame for visualization
        annotated_frame = draw_detections(frame, all_motorcycle_boxes, all_helmet_boxes)
        
        # Detect violations
        violations = []
        
        for motorcycle_box, motorcycle_conf, _ in all_motorcycle_boxes:
            riders = []
            
            for helmet_box, helmet_conf, helmet_label in all_helmet_boxes:
                if is_head_above_motorcycle(helmet_box, motorcycle_box):
                    riders.append((helmet_box, helmet_label, helmet_conf))
            
            # Check for violations
            if riders:
                riders_without_helmet = [r for r in riders if r[1] == "no-helmet"]
                
                if riders_without_helmet or len(riders) > 2:
                    # Extract number plate only when violation is detected
                    number_plate = get_number_plate(motorcycle_box, frame, all_other_boxes)
                    
                    violation_info = {
                        'frame_number': frame_count,
                        'timestamp': f"{int(frame_count//fps):02d}:{int(frame_count%fps):02d}",
                        'motorcycle_confidence': float(motorcycle_conf),
                        'number_plate': number_plate,
                        'total_riders': len(riders),
                        'riders_without_helmet': len(riders_without_helmet),
                        'violations': [],
                        'annotated_frame': annotated_frame
                    }
                    
                    if riders_without_helmet:
                        violation_info['violations'].append(f"No Helmet ({len(riders_without_helmet)} rider(s))")
                    
                    if len(riders) > 2:
                        violation_info['violations'].append(f"Triple Seat ({len(riders)} people)")
                    
                    violations.append(violation_info)
        
        return violations
        
    except Exception as e:
        st.error(f"❌ Frame analysis error: {e}")
        return []

def process_video_with_number_plate(video_path, process_every_n_frames=30):
    """Process video with number plate extraction for violations"""
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("❌ Error: Could not open video file")
            return [], []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            st.warning("⚠️ Could not determine total frames. Processing limited frames.")
            total_frames = 1000
        
        st.info(f"📹 Video info: {total_frames} frames, {fps:.1f} FPS")
        
        frame_count = 0
        all_violations = []
        violation_frames = []
        
        # Create progress elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        violation_counter = st.empty()
        
        start_time = time.time()
        max_frames = min(total_frames, 500)  # Reduced for cloud performance
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % process_every_n_frames == 0:
                # Update progress
                progress = min(frame_count / max_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"📊 Processing frame {frame_count}/{max_frames}")
                violation_counter.text(f"🚨 Violations found: {len(all_violations)}")
                
                # Analyze frame with number plate extraction
                frame_violations = analyze_frame_with_number_plate(
                    frame, frame_count, 
                    st.session_state.custom_model, 
                    st.session_state.coco_model, 
                    fps
                )
                
                # Store violations and frames
                for violation in frame_violations:
                    all_violations.append(violation)
                    
                    # Convert annotated frame to display format
                    annotated_rgb = cv2.cvtColor(violation['annotated_frame'], cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(annotated_rgb)
                    
                    # Resize for display
                    max_size = (400, 300)
                    pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
                    
                    violation_frames.append({
                        'frame_number': violation['frame_number'],
                        'timestamp': violation['timestamp'],
                        'image': pil_image,
                        'violations': violation['violations'],
                        'number_plate': violation['number_plate']
                    })
                
                # Show real-time updates in sidebar
                if frame_violations:
                    for violation in frame_violations[-3:]:  # Show last 3 violations
                        with st.sidebar:
                            st.markdown(f"""
                            <div class="violation-card">
                                <strong>🚨 Violation Detected</strong><br>
                                Frame: {violation['frame_number']}<br>
                                Time: {violation['timestamp']}<br>
                                Number Plate: {violation['number_plate']}<br>
                                Violations: {', '.join(violation['violations'])}
                            </div>
                            """, unsafe_allow_html=True)
            
            frame_count += 1
        
        cap.release()
        progress_bar.progress(1.0)
        
        processing_time = time.time() - start_time
        status_text.text(f"✅ Processing completed in {processing_time:.1f}s")
        violation_counter.text(f"🎯 Total violations found: {len(all_violations)}")
        
        return all_violations, violation_frames
        
    except Exception as e:
        st.error(f"❌ Video processing error: {e}")
        return [], []

# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header">🚨 Helmet Violation Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Groq API status
    if client:
        # st.sidebar.success("✅ Groq API: Connected")
        st.sidebar.success("✅ API: Connected")
    else:
        # st.sidebar.warning("⚠️ Groq API: Not configured")
        st.sidebar.warning("⚠️ API: Not configured")
        # st.sidebar.info("Add GROQ_API_KEY to .env file or Streamlit secrets")
        st.sidebar.info("Add API_KEY to file or Streamlit ")
    # Model loading section
    st.sidebar.markdown("### Model Status")
    
    if not st.session_state.models_loaded:
        if st.sidebar.button("🔄 Load Models", type="primary"):
            with st.spinner("Loading models..."):
                success = load_models()
                if success:
                    st.sidebar.success("✅ Models loaded!")
                else:
                    st.sidebar.error("❌ Failed to load models")
    else:
        st.sidebar.success("✅ Models are loaded and ready!")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload Video File", 
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Supported formats: MP4, AVI, MOV, MKV. Max 200MB recommended."
    )
    
    # Processing parameters
    st.sidebar.markdown("### Processing Settings")
    process_every_n_frames = st.sidebar.slider(
        "Process every N frames",
        min_value=10,
        max_value=50,
        value=20,
        help="Higher values = faster processing but may miss violations"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="sub-header">🎥 Video Analysis</div>', unsafe_allow_html=True)
        
        if not st.session_state.models_loaded:
            st.warning("⚠️ Please load models first using the button in the sidebar")
        
        elif uploaded_file is not None:
            # Display file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
                "File type": uploaded_file.type
            }
            
            st.write("**File Details:**")
            st.json(file_details)
            
            # Display video
            st.video(uploaded_file)
            
            # Process button
            if st.button("🚀 Start Violation Detection", type="primary", use_container_width=True):
                if not st.session_state.models_loaded:
                    st.error("❌ Models not loaded. Please load models first.")
                    return
                
                # Check file size (limit to 200MB for cloud)
                if uploaded_file.size > 200 * 1024 * 1024:
                    st.error("❌ File too large. Please upload a video smaller than 200MB.")
                    return
                
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    video_path = tmp_file.name
                
                st.session_state.processing = True
                
                try:
                    with st.spinner("🔍 Analyzing video for violations..."):
                        violations, violation_frames = process_video_with_number_plate(video_path, process_every_n_frames)
                        
                        st.session_state.results = violations
                        st.session_state.violation_frames = violation_frames
                        
                        # Create DataFrame for results
                        if violations:
                            df_data = []
                            for violation in violations:
                                # Create violation message with vehicle number
                                violation_message = f"{', '.join(violation['violations'])} - Vehicle: {violation['number_plate']}"
                                
                                df_data.append({
                                    'Frame': violation['frame_number'],
                                    'Timestamp': violation['timestamp'],
                                    'Number Plate': violation['number_plate'],
                                    'Violations': ', '.join(violation['violations']),
                                    'Violation Message': violation_message,
                                    'Total Riders': violation['total_riders'],
                                    'Riders without Helmet': violation['riders_without_helmet'],
                                    'Confidence': f"{violation['motorcycle_confidence']:.2f}"
                                })
                            st.session_state.violations_df = pd.DataFrame(df_data)
                        else:
                            st.session_state.violations_df = pd.DataFrame()
                    
                    if violations:
                        st.success(f"✅ Analysis completed! Found {len(violations)} violations.")
                    else:
                        st.info("ℹ️ No violations detected in the video.")
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {e}")
                
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(video_path)
                    except:
                        pass
                    st.session_state.processing = False
        
        else:
            # Demo section when no file is uploaded
            st.markdown("""
            <div class="info-card">
            <h3>👆 How to Use</h3>
            <ol>
                <li><strong>Load Models</strong> using the button in sidebar</li>
                <li><strong>Upload</strong> a video file</li>
                <li><strong>Configure</strong> processing settings</li>
                <li><strong>Click</strong> 'Start Violation Detection'</li>
                <li><strong>View</strong> real-time results and download reports</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            ### 🎯 Detection Capabilities
            - 🏍️ **Motorcycle detection** using YOLO models
            - ⛑️ **Helmet/No-helmet classification**  
            - 🔢 **Number plate extraction** (with Groq API)
            - 🚨 **Violation detection**:
              - No helmet usage
              - Triple seating
            
            ### ⚠️ Cloud Limitations
            - Maximum video size: 200MB
            - Processing limited to 500 frames
            - Models auto-download on first run
            """)
    
    with col2:
        st.markdown('<div class="sub-header">📊 Statistics</div>', unsafe_allow_html=True)
        
        if st.session_state.results is not None:
            violations = st.session_state.results
            
            # Metrics
            st.metric("Total Violations", len(violations))
            
            if violations:
                no_helmet_count = sum(v['riders_without_helmet'] for v in violations)
                triple_seat_count = sum(1 for v in violations if any("Triple Seat" in viol for viol in v['violations']))
                
                # Count readable number plates
                readable_plates = sum(1 for v in violations if v['number_plate'] not in ["No plate detected", "Extraction failed", "API key missing", "Not readable"])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("No Helmet Cases", no_helmet_count)
                with col2:
                    st.metric("Triple Seat Cases", triple_seat_count)
                
                st.metric("Readable Number Plates", readable_plates)
                
                # Violation types chart
                violation_types = {}
                for violation in violations:
                    for v_type in violation['violations']:
                        violation_types[v_type] = violation_types.get(v_type, 0) + 1
                
                if violation_types:
                    fig = px.pie(
                        values=list(violation_types.values()),
                        names=list(violation_types.keys()),
                        title="Violation Types"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No violations detected")
        
        else:
            st.info("📈 Statistics will appear here after processing")
    
    # Results Section
    if st.session_state.violations_df is not None:
        if not st.session_state.violations_df.empty:
            st.markdown('<div class="sub-header">📋 Detection Results</div>', unsafe_allow_html=True)
            
            # Display dataframe
            st.dataframe(st.session_state.violations_df, use_container_width=True)
            
            # Show violation frames
            st.markdown('<div class="sub-header">🖼️ Violation Frames</div>', unsafe_allow_html=True)
            
            if st.session_state.violation_frames:
                # Display frames in columns
                cols = st.columns(2)
                for idx, frame_data in enumerate(st.session_state.violation_frames):
                    col_idx = idx % 2
                    with cols[col_idx]:
                        st.markdown(f"<div class='frame-container'>", unsafe_allow_html=True)
                        st.image(frame_data['image'], 
                                caption=f"Frame {frame_data['frame_number']} | Time: {frame_data['timestamp']} | Plate: {frame_data['number_plate']} | Violations: {', '.join(frame_data['violations'])}",
                                use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No violation frames captured")
            
            # Download buttons
            st.markdown("### 💾 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                # Download as CSV
                csv = st.session_state.violations_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Download as JSON file with violation messages and vehicle numbers
                json_data = []
                for _, row in st.session_state.violations_df.iterrows():
                    json_data.append({
                        "violation_msg": row['Violations'],
                        "vehicle_no": row['Number Plate']
                    })
                
                json_string = json.dumps(json_data, indent=2)
                
                st.download_button(
                    label="📥 Download JSON Report",
                    data=json_string,
                    file_name=f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        else:
            st.success("✅ No violations detected in the video! 🎉")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Helmet Violation Detection System** | "
        "Built with Streamlit, YOLO & Groq API | "
        "Deployed on Streamlit Cloud 🚀"
    )

if __name__ == "__main__":
    main()