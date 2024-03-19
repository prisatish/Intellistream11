import subprocess
import streamlink
import streamlit as st
import tempfile
import base64
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO  
from openai import OpenAI
import whisper
from google.cloud import vision

# st.set_page_config(layout="wide")

load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
if not OpenAI.api_key:     
    raise ValueError("The OpenAI API key must be set in the OPENAI_API_KEY environment variable.")

whisper.api_key = os.getenv("WHISPER_API_KEY")
if not whisper.api_key:
    raise ValueError("The WHsiper API Key needs to be set in the env")
client = OpenAI()

# Set Google Cloud credentials in environment
service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'trueinfolabs-ocr-20c8c095084b.json'

# Initialize Google Vision client
vision_client = vision.ImageAnnotatorClient()

wipro_logo_path = "Wiprologo.jpg"  # Update this path to where your logo is stored
wipro_logo = Image.open(wipro_logo_path)
# Create a layout with columns
col1, col2 = st.columns([8, 2])  # Adjust the ratio as needed

# Display the "Insightly Video" text in the first column (larger space)

# Display the logo in the second column (right side, smaller space)
with col2:
    st.image(wipro_logo, width=200)  # Adjust the width as needed

# Function to execute FFmpeg command and capture output
def execute_ffmpeg_command(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("FFmpeg command executed successfully.")
            return result.stdout, None  # Return None for stderr on success
        else:
            error_message = result.stderr.decode('utf-8')
            print(f"Error executing FFmpeg command: {error_message}")
            return None, error_message  # Return decoded error message
    except Exception as e:
        print(f"An error occurred during FFmpeg execution: {e}")
        return None, str(e)  # Return the exception message as the error
    

# Function to get transcript from audio using OpenAI Whisper
def get_transcript_from_audio(audio_file_path):
    try:
        # Load the model
        model = whisper.load_model("base")  # You can choose another model size if needed
        
        # Process the audio file and get the result
        result = model.transcribe(audio_file_path)
        
        # Get the transcript text
        transcript_text = result["text"]
        return transcript_text
    except Exception as e:
        print(f"Error submitting transcription job: {e}")
        return None

def extract_text_from_base64_frame(base64_frame):
    """Extracts text from a single base64 encoded frame using Google Cloud Vision API."""
    frame_bytes = base64.b64decode(base64_frame)  # Decode the base64 string to bytes
    image = vision.Image(content=frame_bytes)
    response = vision_client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description.strip() if texts else "No text found."

def transcribe_uploaded_mp3(uploaded_mp3):
    try:
        # Save the uploaded MP3 file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            tmpfile.write(uploaded_mp3.getvalue())
            audio_file_path = tmpfile.name
 
        # You might want to process/convert the MP3 file with FFmpeg here if needed
        # For example, to ensure it's in the correct format for Whisper or to extract a specific part
        # This is optional and depends on your requirements
 
        # Transcribe the audio file using Whisper
        transcript = get_transcript_from_audio(audio_file_path)
       
        if transcript is None:
            return "Transcription failed or no transcript available."
       
        return transcript
    except Exception as e:
        return f"Failed to transcribe audio file. Error: {e}"

def analyze_image_with_google_vision_api(base64_frame):
    """Analyze image content using Google Cloud Vision API."""
    frame_bytes = base64.b64decode(base64_frame)  # Decode the base64 string to bytes
    image = vision.Image(content=frame_bytes)

    response = vision_client.label_detection(image=image)
    labels = response.label_annotations

    if labels:
        return ', '.join([label.description for label in labels])
    else:
        return "No labels detected."

def analyze_content_with_openai(text, description,labels):
    """Analyze the combined text and image labels to categorize the image using OpenAI."""
    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": "Classify the following image into 'none' or one of the other\
                  categories based on the extracted text and description of the frames: Bullying, Nudity & \
                 Adult Content, Graphic Violence, Illegal Goods, Child Safety, Sexual Abuse, Profanity, \
                 Suicide/Self Harm, Violent Extremism and None. Return the following: Give out the result\
                  as Category - {whatever the category is} and then GIVE A PROPER JUSTIFICATION of the\
                  image categorization for that conclusion without any assumption from your side. Do not assume\
                 the reasoning behind the content in the frames. "},
                {"role": "user", "content": f"Text: {text}\nDescription: {description}"}
            ],
            max_tokens=4096, 
            n=1
        )
        if response.choices:
            result_message = response.choices[0].message.content
            return result_message.strip()
        else:
            return "Analysis failed or was inconclusive."
    except Exception as e:
        return f"Failed to analyze content with OpenAI. Error: {e}"

def display_categories(analysis_result, categories):
    """Display categories with highlight based on analysis result."""
    # Extracting the 'xyz' from the analysis_result
    try:
        extracted_text = analysis_result.split('Category - ')[1].split('\n')[0].strip()
        category_keywords = [x.strip() for x in extracted_text.split(',')]
    except IndexError:
        # Default to None if parsing fails
        category_keyword = 'None'
    
    num_cols = 3
    rows = [categories[i:i + num_cols] for i in range(0, len(categories), num_cols)]
 
    matched_style = """
        border: 2px solid #00FF00;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        background-color: #333333;
        color: #FFFFFF;
        margin: 5px;
        box-shadow: 0 2px 4px 0 rgba(255,255,255,0.2);
    """
    unmatched_style = """
        border: 1px solid #555555;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        background-color: #222222;
        color: #AAAAAA;
        margin: 5px;
    """
 
    # Display categories in a grid layout
    for row in rows:
        cols = st.columns(num_cols)
        for idx, category in enumerate(row):
            with cols[idx]:
                # Check if the category matches any in the list of extracted categories
                if category.lower() in [k.lower() for k in category_keywords]:
                    # Highlight matched category
                    st.markdown(f"<div style='{matched_style}'><h4 style='margin:0;'>{category}</h4></div>", unsafe_allow_html=True)
                else:
                    # Display non-matched category
                    st.markdown(f"<div style='{unmatched_style}'><h4 style='margin:0;'>{category}</h4></div>", unsafe_allow_html=True)

def execute_fmpeg_command(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        return result.stdout  # Return just the stdout part, not a tuple
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg command failed with error: {e.stderr.decode()}")
        return None

def search_keyword(keyword, frame_texts):
    return [index for index, text in st.session_state.frame_texts.items() if keyword.lower() in text.lower()]

# Function to generate description for video frames
def generate_description(base64_frames):
    try:
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    "1. Generate a description for this sequence of video frames in about 90 words. 2.Return the following: i. List of objects in the video ii. Any restrictive content or sensitive content and if so which frame. iii. The frames is supposed to contain news content and we want to detect non-news content such as an advertisement. So analyze specifically for any indications that the content might be promotional or an advertisement.",
                    *map(lambda x: {"image": x, "resize": 428}, base64_frames),
                ],
            },
        ]
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=prompt_messages,
            max_tokens=3000,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in generate_description: {e}")
        return None

def generate_overall_description(transcript_text, video_description):
    try:
        combined_input = f"Transcript: {transcript_text}\n\nVideo Description: {video_description}\n\n"
        prompt_message = "Based on the above transcript and video description, generate a very detailed description about the sequence of events in the video and from the transcript within 500 words."
        
        prompt_messages = [
            {"role": "user", "content": combined_input + prompt_message}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=prompt_messages,
            max_tokens=1000,  # Increased from 300 to allow for a more detailed response
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in generate_overall_description: {e}")
        return None

with col1:
    is_logo_path = "IntelliStreamLogo.png"  # Update this path to where your logo is stored
    is_logo = Image.open(is_logo_path)
    st.image(is_logo, width=200)

    st.markdown("<h1 style='text-align: left; color: white;'></h1>", unsafe_allow_html=True)

# Streamlit UI
 
    st.title("Insightly Video")
    stream_url = st.text_input("Enter the live stream URL (YouTube, Twitch, etc.):")
    keyword = st.text_input("Enter a keyword to filter the frames (optional):")
    extract_frames_button = st.button("Extract Frames")
    uploaded_video = st.file_uploader("Or upload a video file (MP4):", type=["mp4"])


    # Slider to select the number of seconds for extraction
    seconds = st.slider("Select the number of seconds for extraction:", min_value=1, max_value=60, value=10)

    uploaded_mp3 = st.file_uploader("Upload an MP3 file for transcription:", type=["mp3"])
    
    
    # Check if an MP3 file has been uploaded
    if uploaded_mp3 is not None:
        # Call the transcription function with the uploaded MP3 file
        transcript = transcribe_uploaded_mp3(uploaded_mp3)
    
        # Display the transcript
        st.text_area("Transcript:", value=transcript, height=300)
    else:
        st.write("Please upload an MP3 file to get started.")

    if (extract_frames_button and stream_url and keyword) or (extract_frames_button and stream_url):
    # Execute FFmpeg command to extract frames

    # Check if URL is provided

            streams = streamlink.streams(stream_url)
            if "best" in streams:
                stream_url = streams["best"].url

                ffmpeg_command = [
                'ffmpeg',          # Input stream URL
                '-t', str(seconds),         # Duration to process the input (selected seconds)
                '-vf', 'fps=1',             # Extract one frame per second
                '-f', 'image2pipe',         # Output format as image2pipe
                '-c:v', 'mjpeg',            # Codec for output video
                '-an',                      # No audio
                '-'
            ]
                
            # Determine the input source for FFmpeg
            input_source = stream_url  # Default to stream URL

            # Insert the input source into the FFmpeg command
            ffmpeg_command.insert(1, input_source)
            ffmpeg_command.insert(1, '-i')

            # Execute FFmpeg command
            ffmpeg_output, _ = execute_ffmpeg_command(ffmpeg_command)

        # Modify the section where you display frames to include text extraction and display
        # Modify the section where base64 encoded frames are processed
        # After successfully executing the FFmpeg command to capture frames
            if ffmpeg_output:
                st.write("Frames Extracted:")
                frame_bytes_list = ffmpeg_output.split(b'\xff\xd8')[1:]  # Correct splitting for JPEG frames
                n_frames = len(frame_bytes_list)
                base64_frames = [base64.b64encode(b'\xff\xd8' + frame_bytes).decode('utf-8') for frame_bytes in frame_bytes_list]

                categories_results = []
                frame_texts = {}

                for idx, frame_base64 in enumerate(base64_frames):
                    extracted_text = extract_text_from_base64_frame(frame_base64)
                    frame_texts[idx] = extracted_text

                    if not keyword or keyword.lower() in extracted_text.lower():
                        col1, col2 = st.columns([3, 2])
                        with col1:
                            frame_bytes = base64.b64decode(frame_base64)
                            st.image(Image.open(BytesIO(frame_bytes)), caption=f'Frame {idx + 1}', use_column_width=True)
                        with col2:
                            st.write(f"Extracted Text: {extracted_text}")
                            if keyword:
                                st.write(f"Displaying frames containing the keyword '{keyword}'.")
                            else:
                                st.write("Displaying all extracted frames.")
                    # Use Streamlit columns for side-by-side display (1 column for image, 1 for text)
                  #  col1, col2 = st.columns([3, 2])
                  #  with col1:
                   #     frame_bytes = base64.b64decode(frame_base64)
                   #     st.image(Image.open(BytesIO(frame_bytes)), caption=f'Frame {idx + 1}', use_column_width=True)
                   # with col2:
                   #     st.write(f"Extracted Text: {extracted_text}")

                #    if 'base64_frames' not in st.session_state:
                #        st.session_state.base64_frames = []  # Populate this when frames are first extracted
                #    if 'frame_texts' not in st.session_state:
                #        st.session_state.frame_texts = {}  

                st.write("Analysis Results for All Frames:")
                # Assuming 'categories' is defined with all possible categories you're interested in
                categories = ["Bullying", "Nudity & Adult Content", "Graphic Violence", "Illegal Goods", "Child Safety", "Sexual Abuse", "Profanity", "Self Harm/Suicide", "Violent Extremism","None"]
                # Here, you might want to process combined_analysis_results to summarize or just display them
    
            #    display_categories(" ".join(categories_results), categories)
        
                # Extract audio
            audio_command = [
                'ffmpeg',
                '-i', stream_url,           # Input stream URL
                '-vn',                      # Ignore the video for the audio output
                '-acodec', 'libmp3lame',    # Set the audio codec to MP3
                '-t', str(seconds),         # Duration for the audio extraction (selected seconds)
                '-f', 'mp3',                # Output format as MP3
                '-'
            ]
            audio_output, _ = execute_ffmpeg_command(audio_command)

            st.write("Extracted Audio:")
            audio_tempfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            audio_tempfile.write(audio_output)
            audio_tempfile.close()

            st.audio(audio_output, format='audio/mpeg', start_time=0)

            # Get the transcript from whisper
            transcript_text = get_transcript_from_audio(audio_tempfile.name)
            if transcript_text:
                st.markdown("**Transcript:**")
                st.write(transcript_text)
            else:
                st.write("Failed to retrieve transcript.")


        # Get consolidated description for all frames
            if ffmpeg_output:
                description = generate_description(base64_frames)
                if description:
                    st.markdown("**Frame Description:**")
                    st.write(description)
                else:
                    st.write("Failed to generate description.")

            image_labels = analyze_image_with_google_vision_api(frame_base64)
        #   st.write(image_labels)
            analysis_result = analyze_content_with_openai(extracted_text, description, image_labels)
            st.write(analysis_result)
            display_categories(analysis_result, categories)
            categories_results.append(analysis_result)  # Collect results for summary

            # Get the transcript from whisper
            transcript_text = get_transcript_from_audio(audio_tempfile.name)  
            description = generate_description(base64_frames)
        # Generate overall description using transcript and video description
            overall_description = generate_overall_description(transcript_text, description)
            if overall_description:
                st.markdown("**Consolidated Description:**")
                st.write(overall_description)
            else:
                st.write("Failed to generate overall description.")

            if keyword:
                st.write(f"Displaying frames containing the keyword '{keyword}'.")
            else:
                st.write("Displaying all extracted frames.")

    elif uploaded_video is not None and extract_frames_button:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            tmpfile.write(uploaded_video.getvalue())
            video_file_path = tmpfile.name

            ffmpeg_command = [
                'ffmpeg',          # Input stream URL
                '-i', video_file_path, 
                '-t', str(seconds),          # Duration to process the input (selected seconds)
                '-vf', 'fps=1',             # Extract one frame per second
                '-f', 'image2pipe',         # Output format as image2pipe
                '-c:v', 'mjpeg',            # Codec for output video
                '-an',                      # No audio
                '-'
            ]

            ffmpeg_output = execute_fmpeg_command(ffmpeg_command)

            if ffmpeg_output:
                st.write("Frames Extracted:")
                frame_bytes_list = ffmpeg_output.split(b'\xff\xd8')[1:]  # Correct splitting for JPEG frames
                n_frames = len(frame_bytes_list)
                base64_frames = [base64.b64encode(b'\xff\xd8' + frame_bytes).decode('utf-8') for frame_bytes in frame_bytes_list]

                categories_results = []
                frame_texts = {}

                for idx, frame_base64 in enumerate(base64_frames):
                    extracted_text = extract_text_from_base64_frame(frame_base64)
                    frame_texts[idx] = extracted_text
                    # Use Streamlit columns for side-by-side display (1 column for image, 1 for text)
                    col1, col2 = st.columns([3, 2])
                    with col1:
                        frame_bytes = base64.b64decode(frame_base64)
                        st.image(Image.open(BytesIO(frame_bytes)), caption=f'Frame {idx + 1}', use_column_width=True)
                    with col2:
                        st.write(f"Extracted Text: {extracted_text}")

                
                st.write("Analysis Results for All Frames:")
                # Assuming 'categories' is defined with all possible categories you're interested in
                categories = ["Bullying", "Nudity & Adult Content", "Graphic Violence", "Illegal Goods", "Child Safety", "Sexual Abuse", "Profanity", "Self Harm/Suicide", "Violent Extremism", "None"]
                # Here, you might want to process combined_analysis_results to summarize or just display them
    
                

    
            # Extract audio
            audio_command = [
                'ffmpeg',
                '-i', video_file_path,  
                '-t', str(seconds), 
                '-vf', 'fps=1',         # Input stream URL
                '-vn',                      # Ignore the video for the audio output
                '-acodec', 'libmp3lame',    # Set the audio codec to MP3        # Duration for the audio extraction (selected seconds)
                '-f', 'mp3',                # Output format as MP3
                '-'
            ]
            audio_output, _ = execute_ffmpeg_command(audio_command)

            st.write("Extracted Audio:")
            audio_tempfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            audio_tempfile.write(audio_output)
            audio_tempfile.close()

            st.audio(audio_output, format='audio/mpeg', start_time=0)

            # Get the transcript from whisper
            transcript_text = get_transcript_from_audio(audio_tempfile.name)
            if transcript_text:
                st.markdown("**Transcript:**")
                st.write(transcript_text)
            else:
                st.write("Failed to retrieve transcript.")

                # Get consolidated description for all frames
            if ffmpeg_output:
                description = generate_description(base64_frames)
                if description:
                    st.markdown("**Frame Description:**")
                    st.write(description)
                else:
                    st.write("Failed to generate description.")

            image_labels = analyze_image_with_google_vision_api(frame_base64)
        #  st.write(image_labels)
            analysis_result = analyze_content_with_openai(extracted_text, description, image_labels)
            st.write(analysis_result)
            display_categories(analysis_result, categories)
            categories_results.append(analysis_result)  # Collect results for summary

            # if st.button("Overall Description"):  
            #     audio_tempfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            #     audio_tempfile.write(audio_output)
            #     audio_tempfile.close()

                    # Get the transcript from whisper
            transcript_text = get_transcript_from_audio(audio_tempfile.name)  
            description = generate_description(base64_frames)
        # Generate overall description using transcript and video description
            overall_description = generate_overall_description(transcript_text, description)
            if overall_description:
                st.markdown("**Consolidated Description:**")
                st.write(overall_description)
            else:
                st.write("Failed to generate overall description.")
    
    else:
        st.write(" ")    