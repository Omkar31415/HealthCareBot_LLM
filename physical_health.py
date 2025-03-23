import streamlit as st
import tempfile
import os
import librosa
from transformers import pipeline
from dotenv import load_dotenv
import json
import requests
import anthropic

# Load environment variables
load_dotenv()
# Load secrets
hf_token = st.secrets["HUGGINGFACE_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]
anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]

def run_physical_health():
    # Initialize session state variables
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'diagnostic_state' not in st.session_state:
        st.session_state.diagnostic_state = "initial"  # States: initial, gathering, requesting_audio, diagnosing, complete
    if 'gathered_symptoms' not in st.session_state:
        st.session_state.gathered_symptoms = []
    if 'audio_analysis' not in st.session_state:
        st.session_state.audio_analysis = None
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    if 'current_options' not in st.session_state:
        st.session_state.current_options = None
    if 'llm_service' not in st.session_state:
        st.session_state.llm_service = "groq"  # Default LLM service
    if 'user_submitted' not in st.session_state:
        st.session_state.user_submitted = False
    if 'severity_rating' not in st.session_state:
        st.session_state.severity_rating = None
    if 'audio_processed' not in st.session_state:
        st.session_state.audio_processed = False
    if 'waiting_for_next_question' not in st.session_state:
        st.session_state.waiting_for_next_question = False
    if 'condition_evidence' not in st.session_state:
        st.session_state.condition_evidence = {}
    if 'other_selected' not in st.session_state:
        st.session_state.other_selected = False

    # Callback to handle option selection
    def handle_option_select():
        selected_option = st.session_state.selected_option
        if selected_option:
            # Add selected option to conversation
            if selected_option == "Other":
                st.session_state.other_selected = True
            else:
                # Record the response in conversation
                st.session_state.conversation.append(selected_option)
                st.session_state.gathered_symptoms.append(selected_option)
                
                # Update evidence for conditions
                update_condition_evidence(selected_option)
                
                st.session_state.user_submitted = True
                st.session_state.other_selected = False

    # Callback to handle custom input submission
    def handle_custom_submit():
        custom_input = st.session_state.custom_input
        if custom_input:
            # Add custom input to conversation
            st.session_state.conversation.append(custom_input)
            st.session_state.gathered_symptoms.append(custom_input)
            st.session_state.user_submitted = True
            st.session_state.custom_input = ""  # Clear the input field
            st.session_state.other_selected = False

    # Function to update evidence scores for different conditions
    def update_condition_evidence(selected_option):
        # This serves as a starting point but won't limit the conditions
        if not hasattr(st.session_state, 'condition_symptoms'):
            st.session_state.condition_symptoms = {
            "Upper Respiratory Infection": [
                "runny nose", "congestion", "sore throat", "sneezing", 
                "mild fever", "cough", "headache", "nasal", "sinus"
            ],
            "Bronchitis": [
                "persistent cough", "chest tightness", "shortness of breath", 
                "wheezing", "fatigue", "yellow", "green", "sputum", "phlegm"
            ],
            "Pneumonia": [
                "high fever", "severe cough", "difficulty breathing", "chest pain", 
                "rapid breathing", "rust colored", "blood", "sputum", "chills"
            ],
            "Asthma": [
                "wheezing", "shortness of breath", "chest tightness", 
                "coughing", "difficulty sleeping", "allergies", "exercise"
            ],
            "GERD": [
                "heartburn", "regurgitation", "chest pain", "sour taste", 
                "difficulty swallowing", "night cough", "hoarseness", "throat clearing"
            ],
            "Allergies": [
                "sneezing", "itchy eyes", "runny nose", "congestion", 
                "itchy throat", "seasonal", "pet", "food", "rash"
            ],
            "Osteoarthritis": [
                "joint pain", "stiffness", "swelling", "reduced mobility", "morning stiffness",
                "knee", "hip", "joint", "age", "older", "construction", "physical labor",
                "creaking", "grinding", "warmth", "stairs", "walking"
            ],
            "Rheumatoid Arthritis": [
                "joint pain", "symmetrical", "multiple joints", "morning stiffness", 
                "fatigue", "fever", "swelling", "warmth", "autoimmune"
            ],
            "Gout": [
                "sudden pain", "intense pain", "big toe", "red", "swollen", "hot", 
                "tender", "joint", "alcohol", "meat", "seafood"
            ],
            "Meniscus Tear": [
                "knee pain", "swelling", "popping", "locking", "giving way", 
                "inability to straighten", "twist", "injury", "sports"
            ],
            "Tendinitis": [
                "pain", "tenderness", "mild swelling", "warm", "movement pain",
                "repetitive motion", "overuse", "tendon", "wrist", "elbow", "shoulder", "knee", "heel"
            ]
        }
        
        # Initialize condition evidence if not already done
        for condition in st.session_state.condition_symptoms:
            if condition not in st.session_state.condition_evidence:
                st.session_state.condition_evidence[condition] = 0
        
        option_lower = selected_option.lower()
        
        # Track if any matches are found
        matched = False
        
        # Check against existing conditions
        for condition, symptoms in st.session_state.condition_symptoms.items():
            for symptom in symptoms:
                if symptom in option_lower:
                    st.session_state.condition_evidence[condition] += 1
                    matched = True
        
        # Now check for new conditions mentioned directly in the text
        # List of common medical condition indicators
        condition_indicators = ["disease", "syndrome", "disorder", "itis", "infection", "condition", "illness"]
        
        # Check if text contains a likely medical condition name
        potential_new_conditions = []
        words = option_lower.replace(",", " ").replace(".", " ").split()
        
        # Look for condition patterns
        for i, word in enumerate(words):
            # Check for disease indicators
            is_condition = any(indicator in word for indicator in condition_indicators)
            
            # Check for capitalized words that might be condition names
            capitalized = i > 0 and words[i][0].isupper() and not words[i-1].endswith(".")
            
            if is_condition or capitalized:
                # Extract the potential condition name (include surrounding words for context)
                start_idx = max(0, i-2)
                end_idx = min(len(words), i+3)
                potential_condition = " ".join(words[start_idx:end_idx])
                potential_new_conditions.append(potential_condition)
        
        # Also use LLM to extract any mentions of medical conditions
        if len(selected_option) > 15:  # Only for substantial text
            try:
                # Use more focused LLM prompt to extract conditions
                extract_prompt = f"""
                Extract any specific medical conditions or diseases mentioned in this text. 
                Return ONLY the condition names separated by commas, or "none" if no specific 
                conditions are mentioned: "{selected_option}"
                """
                
                if st.session_state.llm_service == "groq":
                    extracted_conditions = use_groq_api(extract_prompt, max_tokens=50)
                else:
                    extracted_conditions = use_claude_api(extract_prompt, max_tokens=50)
                
                # Add these conditions to our potential list
                if extracted_conditions and "none" not in extracted_conditions.lower():
                    for cond in extracted_conditions.split(","):
                        clean_cond = cond.strip()
                        if clean_cond:
                            potential_new_conditions.append(clean_cond)
            except:
                # If there's an error with the API call, continue without it
                pass
        
        # Process the potential conditions
        for potential_condition in set(potential_new_conditions):  # Use set to remove duplicates
            # Clean up the condition name
            clean_condition = potential_condition.strip()
            if len(clean_condition) > 3:  # Avoid very short terms
                # Add as a new condition if not already present
                condition_key = clean_condition.title()  # Capitalize for consistency
                
                if condition_key not in st.session_state.condition_evidence:
                    # Add to evidence with initial score
                    st.session_state.condition_evidence[condition_key] = 1
                    # Create an empty symptom list for this new condition
                    st.session_state.condition_symptoms[condition_key] = []
                    matched = True
                else:
                    # Increment existing condition score
                    st.session_state.condition_evidence[condition_key] += 1
                    matched = True
        
        # If no specific condition matched but we have symptoms, try another pass with general matching
        if not matched and len(selected_option) > 10:
            for condition, symptoms in st.session_state.condition_symptoms.items():
                for symptom in symptoms:
                    for word in symptom.split():
                        if len(word) > 3 and word in option_lower:
                            st.session_state.condition_evidence[condition] += 0.5
                            matched = True
                            break
                    if matched:
                        break

    # Check if audio would be helpful based on symptoms
    def would_audio_help(condition_evidence):
        # Determine if audio would be helpful based on condition evidence
        respiratory_conditions = ["Upper Respiratory Infection", "Bronchitis", "Pneumonia", "Asthma"]
        orthopedic_conditions = ["Osteoarthritis", "Rheumatoid Arthritis", "Gout", "Meniscus Tear", "Tendinitis"]
        
        # Calculate total evidence for respiratory vs. orthopedic conditions
        respiratory_evidence = sum(st.session_state.condition_evidence.get(condition, 0) for condition in respiratory_conditions)
        orthopedic_evidence = sum(st.session_state.condition_evidence.get(condition, 0) for condition in orthopedic_conditions)
        
        # If there's significantly more evidence for orthopedic issues, audio won't help
        if orthopedic_evidence > respiratory_evidence + 2:
            return False
        
        # If there's any significant evidence for respiratory issues, audio may help
        if respiratory_evidence > 1:
            return True
        
        # Default - if we're not sure what the condition is, audio might help
        return respiratory_evidence > 0

    # Load audio classifier model
    @st.cache_resource
    def load_audio_classifier():
        # Audio classifier for cough/breathing analysis
        audio_classifier = pipeline(
            "audio-classification", 
            model="MIT/ast-finetuned-audioset-10-10-0.4593", 
            token=hf_token
        )
        
        return audio_classifier

    # Use Groq API
    def use_groq_api(prompt, max_tokens=500):
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama3-70b-8192",  # Using LLaMA3 70B model
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            st.error(f"Error from Groq API: {response.text}")
            return "Error communicating with the diagnostic model."

    # Use Claude API function
    def use_claude_api(prompt, max_tokens=1000):
        client = anthropic.Client(api_key=anthropic_api_key)
        
        try:
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=max_tokens,
                temperature=0.3,
                system="You are a medical expert providing diagnostic assistance. Focus on identifying potential conditions based on symptoms and providing evidence-based recommendations.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            st.error(f"Error from Claude API: {str(e)}")
            return "Error communicating with the Claude model."
        
    # Function to analyze audio
    def analyze_audio(audio_file, audio_classifier):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.getvalue())
            tmp_path = tmp.name
        
        try:
            # Process audio
            audio, sr = librosa.load(tmp_path, sr=16000)
            result = audio_classifier(tmp_path)
            
            # Map audio classifications to potential medical implications
            medical_context = interpret_audio_results(result)
            
            os.unlink(tmp_path)
            return result, medical_context
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            st.error(f"Error analyzing audio: {str(e)}")
            return None, None

    def interpret_audio_results(audio_results):
        """Convert audio classification results to medically relevant information"""
        medical_interpretations = {
            "Speech": "Voice patterns may indicate respiratory or neurological conditions.",
            "Cough": "Cough patterns can indicate respiratory conditions like bronchitis, pneumonia, or COVID-19.",
            "Wheeze": "Wheezing may indicate asthma, COPD, or bronchitis.",
            "Breathing": "Breathing patterns may indicate respiratory distress or conditions.",
            "Snoring": "May indicate sleep apnea or nasal obstruction.",
            "Sneeze": "May indicate allergies or upper respiratory infections.",
            "Throat clearing": "May indicate postnasal drip, GERD, or throat irritation.",
            "Gasping": "May indicate severe respiratory distress or sleep apnea."
        }
        
        interpretations = []
        for result in audio_results[:3]:  # Focus on top 3 classifications
            label = result['label']
            confidence = result['score']
            
            # Find relevant medical context for any matching keywords
            for key, interpretation in medical_interpretations.items():
                if key.lower() in label.lower():
                    interpretations.append(f"{label} (Confidence: {confidence:.2f}): {interpretation}")
                    break
            else:
                # If no specific medical interpretation, provide a generic one
                interpretations.append(f"{label} (Confidence: {confidence:.2f}): May provide context for diagnosis.")
        
        return interpretations

    # Generate next question and relevant options based on conversation history
    def generate_next_question_with_options(conversation_history, gathered_symptoms, audio_context=None):
        # Combine all information for context
        audio_info = ""
        if audio_context:
            audio_info = "\nAudio analysis detected: " + ", ".join(audio_context)
        
        symptoms_summary = "\n".join(gathered_symptoms) if gathered_symptoms else "No symptoms reported yet."
        
        # Create condition evidence summary
        evidence_summary = ""
        if st.session_state.condition_evidence:
            evidence_list = []
            for condition, score in st.session_state.condition_evidence.items():
                evidence_list.append(f"{condition}: {score}")
            evidence_summary = "Condition evidence scores: " + ", ".join(evidence_list)
        
        # Create a prompt for determining the next question with options
        prompt = f"""You are an expert medical diagnostic assistant gathering information from a patient.
    Current patient information:
    {symptoms_summary}
    {audio_info}
    {evidence_summary}

    Previous conversation:
    {' '.join([f"{'Patient: ' if i%2==0 else 'Doctor: '}{msg}" for i, msg in enumerate(conversation_history)])}

    Based on the information gathered so far, what is the most important follow-up question to ask the patient?
    Also provide 4-5 most likely answer options based on potential conditions.

    Format your response as a JSON object with the following structure:
    {{
    "question": "Your follow-up question here?",
    "options": [
        "Option 1 (specific symptom or detail)",
        "Option 2 (specific symptom or detail)",
        "Option 3 (specific symptom or detail)",
        "Option 4 (specific symptom or detail)",
        "Option 5 (specific symptom or detail)"
    ]
    }}

    Ensure options are specific, clinically relevant to the likely conditions, and help distinguish between possible diagnoses."""

        # Get next question using selected API
        try:
            if st.session_state.llm_service == "groq":
                response = use_groq_api(prompt, max_tokens=500)
            elif st.session_state.llm_service == "claude":  # New condition for Claude
                response = use_claude_api(prompt, max_tokens=500)
            
            # Parse the JSON response
            try:
                # Find the JSON object within the response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    result = json.loads(json_str)
                    
                    # Ensure proper format
                    if "question" in result and "options" in result:
                        # Always add "Other" option
                        if "Other" not in result["options"]:
                            result["options"].append("Other")
                        return result["question"], result["options"]
                
                # Fallback if JSON parsing fails
                return "Can you provide more details about your symptoms?", [
                    "Symptoms are getting worse", 
                    "Symptoms are about the same", 
                    "Symptoms are improving",
                    "New symptoms have appeared",
                    "Other"
                ]
            except json.JSONDecodeError:
                # Fallback for JSON parsing errors
                return "Can you tell me more about your symptoms?", [
                    "Symptoms are mild", 
                    "Symptoms are moderate", 
                    "Symptoms are severe",
                    "Symptoms come and go",
                    "Other"
                ]
        except Exception as e:
            st.error(f"Error generating question: {str(e)}")
            return "How would you describe your symptoms?", [
                "Getting better", 
                "Getting worse", 
                "Staying the same",
                "Fluctuating throughout the day",
                "Other"
            ]

    # Check if more information is needed
    def needs_more_information(conversation_history, gathered_symptoms, condition_evidence):
        # Create a prompt to determine if we need more information
        evidence_summary = ""
        if condition_evidence:
            evidence_list = []
            for condition, score in condition_evidence.items():
                evidence_list.append(f"{condition}: {score}")
            evidence_summary = "Condition evidence scores: " + ", ".join(evidence_list)
        
        prompt = f"""You are an expert medical diagnostic assistant gathering information from a patient.
    Current patient information:
    {' '.join(gathered_symptoms)}
    {evidence_summary}

    Previous conversation:
    {' '.join([f"{'Patient: ' if i%2==0 else 'Doctor: '}{msg}" for i, msg in enumerate(conversation_history)])}

    Based on the information gathered so far, is there enough information to make a preliminary diagnosis?
    Answer with only YES or NO."""

        try:
            if st.session_state.llm_service == "groq":
                result = use_groq_api(prompt, max_tokens=10)
            else:
                result = use_claude_api(prompt, max_tokens=10)
            
            # Clean up the response to get just YES or NO
            result = result.strip().upper()
            if "NO" in result:
                return True  # Need more information
            else:
                return False  # Have enough information
        except Exception as e:
            st.error(f"Error checking information sufficiency: {str(e)}")
            return True  # Default to needing more information

    # Generate audio request prompt
    def generate_audio_request():
        return "To better understand your condition, it would be helpful to analyze your cough or breathing sounds. Could you please upload an audio recording using the upload button in the sidebar? Alternatively, you can skip this step and continue with text-based questions."

    # Add a flag to track if user has declined audio upload
    if 'audio_declined' not in st.session_state:
        st.session_state.audio_declined = False

    def extract_conditions_from_diagnosis(diagnosis_text):
        """Extract condition names from diagnosis to update our condition evidence"""
        if not diagnosis_text:
            return []
            
        # Create a prompt to extract the conditions
        prompt = f"""Extract only the medical conditions (diseases, syndromes, disorders) 
        mentioned in this diagnosis. DO NOT include symptoms, signs, or explanatory text.
        Return ONLY a comma-separated list of legitimate medical condition names without any
        prefacing text or explanation:
        
        {diagnosis_text}"""
        
        try:
            if st.session_state.llm_service == "groq":
                result = use_groq_api(prompt, max_tokens=100)
            else:
                result = use_claude_api(prompt, max_tokens=100)
                
            # Clean up the result to remove any explanatory text
            # Look for the first comma-separated list in the result
            import re
            
            # Remove any explanatory phrases or headers
            cleaned_result = re.sub(r'^.*?([\w\s]+(?:,\s*[\w\s]+)+).*$', r'\1', result, flags=re.DOTALL)
            
            if cleaned_result == result and "," not in result:
                # If regex didn't match, try to extract any medical condition looking phrases
                condition_indicators = ["disease", "syndrome", "infection", "itis", "disorder"]
                potential_conditions = []
                
                for line in result.split('\n'):
                    line = line.strip()
                    # Skip lines that are likely explanatory
                    if line.startswith("Here") or line.startswith("These") or ":" in line:
                        continue
                        
                    # Check for condition indicators
                    if any(indicator in line.lower() for indicator in condition_indicators):
                        potential_conditions.append(line)
                    # Check for capitalized phrases that might be condition names
                    elif len(line) > 0 and line[0].isupper() and len(line.split()) <= 4:
                        potential_conditions.append(line)
                
                if potential_conditions:
                    cleaned_result = ", ".join(potential_conditions)
                else:
                    cleaned_result = result
            
            # Convert to list of condition names and filter out non-conditions
            conditions = []
            common_symptoms = [
                "pain", "ache", "fever", "cough", "sneeze", "wheeze", "breath", 
                "breathing", "shortness", "fatigue", "tired", "dizzy", "nausea",
                "vomit", "diarrhea", "constipation", "rash", "itch", "swelling",
                "tightness", "pressure", "discomfort"
            ]
            
            for cond in cleaned_result.split(","):
                cond = cond.strip()
                # Skip empty strings or very short terms
                if not cond or len(cond) < 4:
                    continue
                    
                # Skip terms that are clearly symptoms, not conditions
                if any(symptom in cond.lower() for symptom in common_symptoms):
                    continue
                    
                # Only add if it looks like a condition
                if len(cond) >= 4:
                    conditions.append(cond)
            
            return conditions
        except Exception as e:
            st.error(f"Error extracting conditions: {str(e)}")
            # Fallback to simpler approach
            common_indicators = ["disease", "syndrome", "infection", "itis", "disorder"]
            words = diagnosis_text.split()
            conditions = []
            
            for i, word in enumerate(words):
                if any(indicator in word.lower() for indicator in common_indicators):
                    start_idx = max(0, i-2)
                    end_idx = min(len(words), i+1)
                    potential_condition = " ".join(words[start_idx:end_idx])
                    conditions.append(potential_condition)
                    
            return conditions
        
    # Generate diagnosis based on all gathered information using selected API
    def generate_diagnosis(conversation_history, gathered_symptoms, condition_evidence, audio_context=None):
        # Combine all information
        audio_info = ""
        if audio_context:
            audio_info = "\nAudio analysis detected: " + ", ".join(audio_context)
        
        symptoms_summary = "\n".join(gathered_symptoms) if gathered_symptoms else "Limited symptom information available."
        
        # Create condition evidence summary with all conditions found
        evidence_summary = ""
        if condition_evidence:
            evidence_list = []
            for condition, score in sorted(condition_evidence.items(), key=lambda x: x[1], reverse=True):
                evidence_list.append(f"{condition}: {score}")
            evidence_summary = "Condition evidence scores: " + ", ".join(evidence_list)
        
        # Create prompt with all potential conditions
        prompt = f"""Act as an expert medical diagnostic assistant. Based on the following patient information, provide:
    1. Potential diagnoses with likelihood assessment (high, moderate, or low probability)
    2. Overall severity rating (High, Moderate, Low) based on the most likely diagnosis
    3. Recommended next steps based on severity:
    - If HIGH severity: Urgently recommend medical attention and specify which specialists to see
    - If MODERATE severity: Recommend medical consultation and provide management tips until seen
    - If LOW severity: Provide self-care tips and when to seek medical attention if symptoms worsen
    4. When the patient should seek immediate medical attention

    Patient information:
    {symptoms_summary}
    {audio_info}
    {evidence_summary}

    Conversation history:
    {' '.join([f"{'Patient: ' if i%2==0 else 'Doctor: '}{msg}" for i, msg in enumerate(conversation_history)])}

    Consider ALL possible relevant conditions, including those not mentioned in the evidence scores.
    Provide a comprehensive but concise diagnostic assessment and recommendations, clearly indicating the SEVERITY RATING (High, Moderate, or Low) at the beginning of your response:"""

        # Rest of the function remains the same
        try:
            if st.session_state.llm_service == "groq":
                diagnosis = use_groq_api(prompt, max_tokens=500)
            else:  # Claude 3 Opus
                diagnosis = use_claude_api(prompt, max_tokens=500)
            
            # Extract severity rating
            severity = None
            if "SEVERITY RATING: HIGH" in diagnosis.upper():
                severity = "High"
            elif "SEVERITY RATING: MODERATE" in diagnosis.upper():
                severity = "Moderate"
            elif "SEVERITY RATING: LOW" in diagnosis.upper():
                severity = "Low"
            
            # Add disclaimer
            diagnosis += "\n\n[Note: This is an AI-generated assessment for testing purposes only and should not replace professional medical advice.]"
            
            return diagnosis, severity
        except Exception as e:
            st.error(f"Error generating diagnosis: {str(e)}")
            return "Error during diagnostic assessment. Please try again.", None

    def display_interactive_diagnosis(diagnosis_text, severity_rating, condition_evidence):
        """Display an interactive, visually appealing summary of the diagnostic assessment using only Streamlit."""
        
        st.markdown("## Your Diagnostic Assessment")
        
        # Create tabs for different sections of the report
        diagnosis_tabs = st.tabs(["Overview", "Conditions", "Recommendations", "Action Steps"])
        
        with diagnosis_tabs[0]:  # Overview tab
            st.markdown("### Assessment Summary")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("#### Key Findings")
                # Simplify diagnosis_text into a summary
                summary_length = 100
                if len(diagnosis_text) > summary_length:
                    summary = diagnosis_text[:summary_length].rsplit(' ', 1)[0] + "..."
                else:
                    summary = diagnosis_text
                st.info(f"Summary: {summary}")
                
                st.markdown("#### Self-Care Recommendations")
                # Fetch self-care recommendations from LLM
                rec_prompt = f"Based on this diagnosis: '{diagnosis_text}', provide 3 concise self-care recommendations."
                if st.session_state.llm_service == "groq":
                    rec_response = use_groq_api(rec_prompt, max_tokens=150)
                else:
                    rec_response = use_claude_api(rec_prompt, max_tokens=150)
                recommendations = rec_response.split("\n")[:3]
                for rec in recommendations:
                    if rec.strip():
                        st.success(f"• {rec.strip()}")

            with col2:
                # Severity status display (kept simple as per your logic)
                if severity_rating == "High":
                    action_needed = "Immediate Medical Attention"
                elif severity_rating == "Moderate":
                    action_needed = "Medical Consultation Recommended"
                else:
                    action_needed = "Follow Self-Care Guidelines"
                
                st.markdown(f"### Severity Level: {severity_rating}")
                st.write(action_needed)
                
                st.markdown("### How are you feeling now?")
                current_feeling = st.select_slider(
                    "My current symptoms are:",
                    options=["Much Worse", "Worse", "Same", "Better", "Much Better"],
                    value="Same"
                )
                
                if current_feeling in ["Much Worse", "Worse"]:
                    st.write("🚨 Consider seeking immediate medical attention if symptoms are worsening.")
                elif current_feeling == "Same":
                    st.write("👨‍⚕️ Follow the recommended care steps and monitor your symptoms.")
                else:
                    st.write("✅ Great! Continue following recommendations for full recovery.")
        
        with diagnosis_tabs[1]:  # Conditions tab
            st.markdown("### Potential Conditions")
            
            if condition_evidence:
                sorted_conditions = sorted(
                    condition_evidence.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
                for condition, score in sorted_conditions:
                    if score > 0:
                        percentage = min(score / 5 * 100, 100)
                        st.write(f"**{condition}**: Probability Score: {score}")
                        st.progress(percentage / 100)
            
            with st.expander("Learn More About These Conditions"):
                if condition_evidence:
                    for condition in condition_evidence.keys():
                        # Fetch condition description from LLM
                        desc_prompt = f"Provide a brief description (1-2 sentences) of the medical condition '{condition}'."
                        if st.session_state.llm_service == "groq":
                            description = use_groq_api(desc_prompt, max_tokens=100)
                        else:
                            description = use_claude_api(desc_prompt, max_tokens=100)
                        st.write(f"**{condition}**: {description.strip()}")
                else:
                    st.write("No specific conditions identified yet.")
        
        with diagnosis_tabs[2]:  # Recommendations tab
            st.markdown("### Care Recommendations")
            
            st.markdown("#### Warning Signs - Seek Medical Help If:")
            # Fetch warning signs from LLM
            warn_prompt = f"Based on this diagnosis: '{diagnosis_text}' and severity '{severity_rating}', list 3 warning signs indicating the need for immediate medical help."
            if st.session_state.llm_service == "groq":
                warn_response = use_groq_api(warn_prompt, max_tokens=150)
            else:
                warn_response = use_claude_api(warn_prompt, max_tokens=150)
            warnings = warn_response.split("\n")[:3]
            for warning in warnings:
                if warning.strip():
                    st.write(f"⚠️ {warning.strip()}")
            
            st.markdown("#### Medications to Consider")
            # Fetch medications from LLM
            med_prompt = f"Based on this diagnosis: '{diagnosis_text}', suggest 3 medications or treatment options."
            if st.session_state.llm_service == "groq":
                med_response = use_groq_api(med_prompt, max_tokens=150)
            else:
                med_response = use_claude_api(med_prompt, max_tokens=150)
            medications = med_response.split("\n")[:3]
            col1, col2 = st.columns(2)
            for i, med in enumerate(medications):
                if med.strip():
                    with col1 if i % 2 == 0 else col2:
                        st.write(f"💊 {med.strip()}")
            
            st.markdown("#### Home Care Tips")
            # Fetch home care tips from LLM
            care_prompt = f"Based on this diagnosis: '{diagnosis_text}', provide 3 home care tips."
            if st.session_state.llm_service == "groq":
                care_response = use_groq_api(care_prompt, max_tokens=150)
            else:
                care_response = use_claude_api(care_prompt, max_tokens=150)
            home_care = care_response.split("\n")[:3]
            for tip in home_care:
                if tip.strip():
                    st.write(f"✅ {tip.strip()}")
        
        with diagnosis_tabs[3]:  # Action Steps tab
            st.markdown("### Next Steps")
            
            if severity_rating in ["High", "Moderate"]:
                st.markdown("#### Medical Consultation")
                # Fetch specialists from LLM
                spec_prompt = f"Based on this diagnosis: '{diagnosis_text}' and conditions: {list(condition_evidence.keys())}, suggest 3 types of medical specialists to consult."
                if st.session_state.llm_service == "groq":
                    spec_response = use_groq_api(spec_prompt, max_tokens=150)
                else:
                    spec_response = use_claude_api(spec_prompt, max_tokens=150)
                specialists = spec_response.split("\n")[:3]
                col1, col2 = st.columns(2)
                for i, spec in enumerate(specialists):
                    if spec.strip():
                        with col1 if i % 2 == 0 else col2:
                            st.write(f"👨‍⚕️ {spec.strip()}")
                
                st.markdown("#### Schedule Consultation")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Find Doctors Near Me", key="find_doctors"):
                        st.info("This would connect to a directory of medical providers.")
                with col2:
                    if st.button("Virtual Consultation Options", key="virtual_consult"):
                        st.info("This would connect to telemedicine services.")
            
            st.markdown("#### Symptom Monitoring")
            with st.expander("Add Symptom Entry"):
                with st.form("symptom_tracker"):
                    st.date_input("Date", value=None)
                    st.slider("Temperature (°F)", min_value=96.0, max_value=104.0, value=98.6, step=0.1)
                    st.slider("Cough Severity", min_value=0, max_value=10, value=5)
                    st.slider("Overall Feeling", min_value=0, max_value=10, value=5)
                    st.form_submit_button("Save Symptom Entry")
        
        st.markdown("---")
        st.write("### Important Note")
        st.write("Monitor your symptoms closely and seek medical attention if they worsen.")
        st.caption("This assessment is for testing and educational purposes only.")
        
        st.markdown("### What would you like to do next?")
        next_steps = st.columns(3)
        
        with next_steps[0]:
            if st.button("Print Assessment", key="print_assessment"):
                # Generate PDF content dynamically
                pdf_content = "Your Diagnostic Assessment\n\n"
                pdf_content += f"Severity Level: {severity_rating}\nAction Needed: {action_needed}\n\n"
                pdf_content += f"Summary: {summary}\n\nConditions:\n"
                if condition_evidence:
                    sorted_conditions = sorted(condition_evidence.items(), key=lambda x: x[1], reverse=True)[:5]
                    for condition, score in sorted_conditions:
                        desc_prompt = f"Provide a brief description of '{condition}'."
                        if st.session_state.llm_service == "groq":
                            description = use_groq_api(desc_prompt, max_tokens=100)
                        else:
                            description = use_claude_api(desc_prompt, max_tokens=100)
                        pdf_content += f"- {condition} (Score: {score}): {description.strip()}\n"
                pdf_content += "\nCare Recommendations:\n"
                pdf_content += "Warning Signs:\n" + "\n".join([f"- {w.strip()}" for w in warnings if w.strip()]) + "\n"
                pdf_content += "Medications:\n" + "\n".join([f"- {m.strip()}" for m in medications if m.strip()]) + "\n"
                pdf_content += "Home Care Tips:\n" + "\n".join([f"- {t.strip()}" for t in home_care if t.strip()]) + "\n"
                pdf_content += "\nFull Diagnosis:\n" + diagnosis_text
                
                # Convert to bytes for download
                pdf_bytes = pdf_content.encode('utf-8')
                st.download_button(
                    label="Download Assessment",
                    data=pdf_bytes,
                    file_name="diagnostic_assessment.txt",  # Using .txt due to Streamlit-only constraint
                    mime="text/plain",
                    key="download_assessment"
                )
                st.success("Assessment ready for download above.")
        
        with next_steps[1]:
            if st.button("Share with Doctor", key="share_doctor"):
                st.success("In a full implementation, this would prepare the assessment for sharing.")
        
        with next_steps[2]:
            if st.button("Start New Consultation", key="start_new"):
                return "new"
        
        return None

    # Main application
    st.title("AI Diagnostic Assistant")
    st.markdown("_This is a prototype for testing purposes only and should not be used for actual medical diagnosis._")
    
    # Side panel for audio upload and conversation controls
    with st.sidebar:
        st.header("Controls")
        
        # LLM Service Selection
        st.subheader("LLM Service")
        llm_option = st.radio(
            "Select LLM Service", 
            ["Groq (LLaMA3-70B)", "Anthropic Claude-3 Opus"],
            index=0 if st.session_state.llm_service == "groq" else 1
        )
        
        # Update LLM service based on selection and reset conversation if service changes
        if (llm_option == "Groq (LLaMA3-70B)" and st.session_state.llm_service != "groq") or \
        (llm_option == "Anthropic Claude-3 Opus" and st.session_state.llm_service != "claude"):
            # Store the new service selection
            st.session_state.llm_service = "groq" if llm_option == "Groq (LLaMA3-70B)" else "claude"
            
            # Reset conversation state
            st.session_state.conversation = []
            st.session_state.diagnostic_state = "initial"
            st.session_state.gathered_symptoms = []
            st.session_state.audio_analysis = None
            st.session_state.current_question = None
            st.session_state.current_options = None
            st.session_state.user_submitted = False
            st.session_state.severity_rating = None
            st.session_state.audio_processed = False
            st.session_state.waiting_for_next_question = False
            st.session_state.condition_evidence = {}
            st.session_state.other_selected = False
            st.session_state.audio_declined = False  # Reset audio declined flag
            
            # Show notification
            st.success(f"Switched to {llm_option}. Starting new consultation.")
            st.rerun()
        
        # Audio upload
        st.subheader("Audio Analysis")
        audio_file = st.file_uploader("Upload audio (cough, breathing, etc.)", type=["wav", "mp3"])
        
        # Handle audio analysis
        if audio_file and not st.session_state.audio_analysis:
            if st.button("Analyze Audio"):
                with st.spinner("Analyzing audio sample..."):
                    try:
                        audio_classifier = load_audio_classifier()
                        audio_results, audio_context = analyze_audio(audio_file, audio_classifier)
                        if audio_results:
                            st.session_state.audio_analysis = audio_context
                            st.session_state.audio_processed = True
                            # If we were in the requesting_audio state, set flag to generate next question
                            if st.session_state.diagnostic_state == "requesting_audio":
                                st.session_state.waiting_for_next_question = True
                            st.success("Audio analysis complete!")
                    except Exception as e:
                        st.error(f"Error analyzing audio: {str(e)}")
        
        # Replace the current condition evidence display in the sidebar with this code
        if st.session_state.condition_evidence:
            st.subheader("Condition Evidence")
            
            # Filter out items that are not likely to be real medical conditions
            non_conditions = ["here is", "here are", "these are", "following", "specific", "mentioned"]
            filtered_evidence = {
                condition: score for condition, score in st.session_state.condition_evidence.items()
                if not any(nc in condition.lower() for nc in non_conditions) and len(condition) > 3
            }
            
            # Create a sorted list of all conditions based on evidence score
            sorted_conditions = sorted(
                filtered_evidence.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Display all conditions with their evidence scores in a cleaner format
            for condition, score in sorted_conditions:
                # Only show conditions with positive scores
                if score > 0:
                    # Calculate percentage (max score assumed to be 5 for full bar)
                    percentage = min(score / 5 * 100, 100)
                    
                    # Create colored bars based on evidence strength
                    if score >= 3:
                        bar_color = "rgba(0, 204, 102, 0.8)"  # Green for strong evidence
                    elif score >= 1.5:
                        bar_color = "rgba(255, 153, 51, 0.8)"  # Orange for moderate evidence
                    else:
                        bar_color = "rgba(160, 160, 160, 0.8)"  # Gray for weak evidence
                    
                    # Display condition with score and bar
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 5px;">
                            <span style="font-size: 0.9em;">{condition}: {score}</span>
                            <div style="background-color: #f0f0f0; height: 10px; border-radius: 5px; margin-top: 2px;">
                                <div style="width: {percentage}%; background-color: {bar_color}; height: 10px; border-radius: 5px;"></div>
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
        
        # Reset conversation button
        st.subheader("Session")
        if st.button("Start New Consultation"):
            st.session_state.conversation = []
            st.session_state.diagnostic_state = "initial"
            st.session_state.gathered_symptoms = []
            st.session_state.audio_analysis = None
            st.session_state.current_question = None
            st.session_state.current_options = None
            st.session_state.user_submitted = False
            st.session_state.severity_rating = None
            st.session_state.audio_processed = False
            st.session_state.waiting_for_next_question = False
            st.session_state.condition_evidence = {}
            st.session_state.other_selected = False
            st.session_state.audio_declined = False  # Reset audio declined flag
            st.rerun()
    
    # Main chat interface
    st.header("Medical Consultation")
    
    # Display selected LLM service
    st.caption(f"Using {'Groq (LLaMA3-70B)' if st.session_state.llm_service == 'groq' else 'Claude 3 Opus'} for diagnosis")
    
    # Display conversation history
    for i, message in enumerate(st.session_state.conversation):
        if i % 2 == 0:  # Assistant messages
            st.markdown(f"**Medical Assistant:** {message}")
        else:  # User messages
            st.markdown(f"**You:** {message}")
    
    # Handle audio processing and next question generation after analysis
    if st.session_state.audio_processed and st.session_state.waiting_for_next_question:
        st.session_state.diagnostic_state = "gathering"  # Resume gathering state
        
        # Generate next question based on all info including audio
        with st.spinner("Processing audio analysis and preparing next question..."):
            question, options = generate_next_question_with_options(
                st.session_state.conversation, 
                st.session_state.gathered_symptoms,
                st.session_state.audio_analysis
            )
            
            # Add the generated question to conversation only if it's not already there
            if len(st.session_state.conversation) == 0 or question != st.session_state.conversation[-1]:
                # Make sure this is an assistant message (should be at an even index)
                if len(st.session_state.conversation) % 2 == 0:
                    st.session_state.conversation.append(question)
            
            st.session_state.current_question = question
            st.session_state.current_options = options
            
            # Reset flags
            st.session_state.waiting_for_next_question = False
            st.session_state.audio_processed = False
            
            # Rerun to display updated conversation
            st.rerun()
    
    # Initial greeting - only show if conversation is empty
    if st.session_state.diagnostic_state == "initial":
        initial_greeting = "Hello, I'm your medical assistant for today. Could you please tell me what symptoms you're experiencing?"
        st.markdown(f"**Medical Assistant:** {initial_greeting}")
        
        # Add a callback for the initial symptoms form submission
        def submit_initial_symptoms():
            if st.session_state.initial_symptoms:
                # Record the user's symptoms in the conversation
                st.session_state.conversation.append(initial_greeting)
                st.session_state.conversation.append(st.session_state.initial_symptoms)
                st.session_state.gathered_symptoms.append(st.session_state.initial_symptoms)
                
                # Update condition evidence based on initial symptoms
                update_condition_evidence(st.session_state.initial_symptoms)
                
                # Change state to indicate we need to generate the next question
                st.session_state.diagnostic_state = "generate_next_question"
        
        # Add an input field for initial symptoms
        with st.form(key="initial_symptoms_form"):
            st.text_area("Please describe your symptoms:", key="initial_symptoms")
            st.form_submit_button("Submit", on_click=submit_initial_symptoms)

    # Add a new state to handle generating the next question after initial symptoms
    elif st.session_state.diagnostic_state == "generate_next_question":
        with st.spinner("Analyzing your symptoms..."):
            question, options = generate_next_question_with_options(
                st.session_state.conversation,
                st.session_state.gathered_symptoms,
                st.session_state.audio_analysis
            )
            
            st.session_state.current_question = question
            st.session_state.current_options = options
            st.session_state.conversation.append(question)
            st.session_state.diagnostic_state = "gathering"
            
            # Force a rerun to refresh the page with the new conversation state
            st.rerun()
    
    # Current assistant question (if in gathering phase)
    elif st.session_state.diagnostic_state == "gathering" and st.session_state.current_question:
        # Only show current question if it's not already in conversation history
        if len(st.session_state.conversation) == 0 or st.session_state.current_question != st.session_state.conversation[-1]:
            # Ensure this is an assistant message (should be at an odd index in conversation)
            if len(st.session_state.conversation) % 2 == 0:
                st.markdown(f"**Medical Assistant:** {st.session_state.current_question}")
    
    # Audio request state
    elif st.session_state.diagnostic_state == "requesting_audio" and not st.session_state.audio_analysis:
        # Only add the audio request to conversation once
        if len(st.session_state.conversation) == 0 or st.session_state.current_question != st.session_state.conversation[-1]:
            # Make sure this is an assistant message (should be at an even index in conversation)
            if len(st.session_state.conversation) % 2 == 0:
                st.session_state.conversation.append(st.session_state.current_question)
        
        # Always display the request
        st.markdown(f"**Medical Assistant:** {st.session_state.current_question}")
        
        # Add buttons to skip or continue with audio upload
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Skip Audio Upload"):
                # Mark audio as declined
                st.session_state.audio_declined = True
                
                # Add user response to conversation
                st.session_state.conversation.append("I'd like to skip the audio upload and continue with text-based questions.")
                
                # Generate the next question and go back to gathering state
                st.session_state.diagnostic_state = "gathering"
                
                # Generate next question
                with st.spinner("Preparing next question..."):
                    question, options = generate_next_question_with_options(
                        st.session_state.conversation,
                        st.session_state.gathered_symptoms,
                        None  # No audio analysis
                    )
                    
                    st.session_state.current_question = question
                    st.session_state.current_options = options
                    st.session_state.conversation.append(question)
                
                st.rerun()
        
        with col2:
            st.info("Please upload an audio sample using the sidebar upload control")
    
    # Display final diagnosis (if complete)
    elif st.session_state.diagnostic_state == "complete":
        if 'diagnosis' in st.session_state:
            action = display_interactive_diagnosis(
                st.session_state.diagnosis,
                st.session_state.severity_rating,
                st.session_state.condition_evidence
            )
            # Handle the "Start New Consultation" action
            if action == "new":
                # Reset session state for a new consultation
                st.session_state.conversation = []
                st.session_state.diagnostic_state = "initial"
                st.session_state.gathered_symptoms = []
                st.session_state.audio_analysis = None
                st.session_state.current_question = None
                st.session_state.current_options = None
                st.session_state.user_submitted = False
                st.session_state.severity_rating = None
                st.session_state.audio_processed = False
                st.session_state.waiting_for_next_question = False
                st.session_state.condition_evidence = {}
                st.session_state.other_selected = False
                st.session_state.audio_declined = False
                st.rerun()  # Rerun the app to reflect the reset state
    
    # Show options for user selection if we have current options and are in gathering state
    if st.session_state.diagnostic_state == "gathering" and st.session_state.current_options and not st.session_state.user_submitted and not st.session_state.other_selected:
        # Create a container for the radio buttons to ensure proper rendering
        options_container = st.container()
        with options_container:
            # Initialize the radio without an on_change callback
            st.radio(
                "Select your response:",
                st.session_state.current_options,
                key="selected_option"
            )
            # Add a submit button to confirm selection
            if st.button("Confirm Selection"):
                if "selected_option" in st.session_state and st.session_state.selected_option:
                    if st.session_state.selected_option == "Other":
                        st.session_state.other_selected = True
                    else:
                        # Record the response in conversation
                        st.session_state.conversation.append(st.session_state.selected_option)
                        st.session_state.gathered_symptoms.append(st.session_state.selected_option)
                        
                        # Update evidence for conditions
                        update_condition_evidence(st.session_state.selected_option)
                        
                        st.session_state.user_submitted = True
                    st.rerun()
            
    # Show custom input field if "Other" is selected
    if st.session_state.other_selected:
        with st.form(key="custom_input_form"):
            st.text_area("Please describe your symptoms in detail:", key="custom_input")
            submit_button = st.form_submit_button("Submit", on_click=handle_custom_submit)
    
    # Process form submission after page rerun
    if st.session_state.user_submitted and st.session_state.diagnostic_state == "gathering":
        # After several questions, check if we need audio data for better diagnosis
        # But only request audio if user hasn't already declined and we don't already have audio
        if len(st.session_state.gathered_symptoms) >= 2 and not st.session_state.audio_analysis and not st.session_state.audio_declined and needs_more_information(
            st.session_state.conversation, 
            st.session_state.gathered_symptoms,
            st.session_state.condition_evidence
        ) and would_audio_help(st.session_state.condition_evidence):
            # Request audio sample
            audio_request = generate_audio_request()
            st.session_state.current_question = audio_request
            st.session_state.conversation.append(audio_request)
            st.session_state.diagnostic_state = "requesting_audio"
        # Determine next action based on conversation length
        elif len(st.session_state.conversation) >= 9 or not needs_more_information(
            st.session_state.conversation, 
            st.session_state.gathered_symptoms,
            st.session_state.condition_evidence
        ):
            # Generate diagnosis after enough information gathered
            with st.spinner("Analyzing your symptoms..."):
                diagnosis, severity = generate_diagnosis(
                    st.session_state.conversation, 
                    st.session_state.gathered_symptoms,
                    st.session_state.condition_evidence,
                    st.session_state.audio_analysis
                )
                
                # Extract conditions from the diagnosis and update evidence
                new_conditions = extract_conditions_from_diagnosis(diagnosis)
                for condition in new_conditions:
                    if condition not in st.session_state.condition_evidence:
                        # Add new conditions from the diagnosis
                        st.session_state.condition_evidence[condition] = 2  # Give it a reasonable starting score
                    else:
                        # Increase evidence for conditions mentioned in diagnosis
                        st.session_state.condition_evidence[condition] += 1

                st.session_state.diagnosis = diagnosis
                st.session_state.severity_rating = severity
                st.session_state.conversation.append("Based on your symptoms, I've prepared a diagnostic assessment.")
                st.session_state.diagnostic_state = "complete"
        else:
            # Generate next question with options
            with st.spinner("Thinking..."):
                question, options = generate_next_question_with_options(
                    st.session_state.conversation, 
                    st.session_state.gathered_symptoms,
                    st.session_state.audio_analysis
                )
                
                st.session_state.current_question = question
                st.session_state.current_options = options
                st.session_state.conversation.append(question)
        
        # Reset user_submitted flag to prepare for next input
        st.session_state.user_submitted = False
        
        # Refresh the page to show the updated conversation
        st.rerun()
    
    # Display audio analysis results if available
    if st.session_state.audio_analysis:
        with st.expander("Audio Analysis Results"):
            st.write("The following was detected in your audio sample:")
            for context in st.session_state.audio_analysis:
                st.write(f"- {context}")

    # Large disclaimer at bottom of app
    st.markdown("---")
    st.warning("⚠️ **IMPORTANT DISCLAIMER**: This application is for testing and educational purposes only. It does not provide actual medical diagnosis or treatment recommendations. Always consult with qualified healthcare professionals for medical advice.")

    # Back button to return to main menu
    if st.button("Back to Main Menu"):
        st.session_state.page = "main"
        st.rerun()

if __name__ == "__main__":
    run_physical_health()