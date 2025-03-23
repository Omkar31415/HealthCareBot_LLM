import streamlit as st
import os
import json
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
#groq_api_key = os.getenv("GROQ_API_KEY")
# Load secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

def run_mental_health():
    # Initialize NLTK resources if not already downloaded
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

    # Initialize session state variables
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'therapy_state' not in st.session_state:
        st.session_state.therapy_state = "initial"  # States: initial, assessment, therapy, summary
    if 'mental_health_scores' not in st.session_state:
        st.session_state.mental_health_scores = {
            "anxiety": 0,
            "depression": 0,
            "stress": 0,
            "loneliness": 0,
            "grief": 0,
            "relationship_issues": 0,
            "self_esteem": 0,
            "trauma": 0
        }
    if 'assessment_progress' not in st.session_state:
        st.session_state.assessment_progress = 0
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    if 'current_options' not in st.session_state:
        st.session_state.current_options = None
    if 'llm_service' not in st.session_state:
        st.session_state.llm_service = "llama3"  # Default LLM service: llama3 or blenderbot
    if 'user_submitted' not in st.session_state:
        st.session_state.user_submitted = False
    if 'severity_rating' not in st.session_state:
        st.session_state.severity_rating = None
    if 'sentiment_analysis' not in st.session_state:
        st.session_state.sentiment_analysis = []
    if 'other_selected' not in st.session_state:
        st.session_state.other_selected = False
    if 'response_delay' not in st.session_state:
        st.session_state.response_delay = False

    # Load Blenderbot model and tokenizer
    @st.cache_resource
    def load_blenderbot_model():
        tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
        return tokenizer, model

    # Load sentiment analyzer
    @st.cache_resource
    def load_sentiment_analyzer():
        return SentimentIntensityAnalyzer()

    # Use Groq API with LLaMA3 model
    def use_llama3_api(prompt, max_tokens=1000):
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama3-70b-8192",  # Using LLaMA3 70B model
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=10  # Add timeout
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                st.error(f"Error from Groq API: {response.text}")
                return "I'm having trouble connecting. Let's continue our conversation more simply."
        except Exception as e:
            st.error(f"Exception when calling Groq API: {str(e)}")
            return "I encountered an error when trying to respond. Let me try a simpler approach."

    # Use Blenderbot for simple responses
    def use_blenderbot(input_text, tokenizer, model):
        try:
            inputs = tokenizer([input_text], return_tensors="pt")
            reply_ids = model.generate(**inputs, max_length=100)
            response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
            return response
        except Exception as e:
            st.error(f"Error from Blenderbot: {str(e)}")
            return "I'm having trouble understanding that. Could you rephrase or tell me more?"

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
                
                # Update mental health scores based on response
                update_mental_health_scores(selected_option)
                
                st.session_state.user_submitted = True
                st.session_state.other_selected = False
                
                # Don't generate options here, let the main loop handle it

    # Callback to handle custom input submission
    def handle_custom_submit():
        custom_input = st.session_state.custom_input
        if custom_input:
            # Add custom input to conversation
            st.session_state.conversation.append(custom_input)
            
            # Update mental health scores based on custom input
            update_mental_health_scores(custom_input)
            
            st.session_state.user_submitted = True
            st.session_state.custom_input = ""  # Clear the input field
            st.session_state.other_selected = False

    # Function to update mental health scores based on user input
    def update_mental_health_scores(user_input):
        # Load sentiment analyzer
        sia = load_sentiment_analyzer()
        sentiment = sia.polarity_scores(user_input)
        
        # Store sentiment analysis for tracking mood over time
        st.session_state.sentiment_analysis.append({
            "text": user_input,
            "sentiment": sentiment,
            "timestamp": time.time()
        })
        
        # Keywords related to different mental health issues
        mental_health_keywords = {
            "anxiety": ["anxious", "nervous", "worry", "panic", "fear", "stress", "tense", "overwhelm", "anxiousness", "uneasy"],
            "depression": ["sad", "depress", "hopeless", "meaningless", "empty", "tired", "exhausted", "unmotivated", "worthless", "guilt"],
            "stress": ["stress", "pressure", "overwhelm", "burden", "strain", "tension", "burnout", "overworked", "deadline", "rush"],
            "loneliness": ["lonely", "alone", "isolate", "disconnect", "abandoned", "reject", "outcast", "friendless", "solitary", "unloved"],
            "grief": ["grief", "loss", "death", "miss", "mourn", "gone", "passed away", "bereavement", "widow", "funeral"],
            "relationship_issues": ["relationship", "partner", "marriage", "divorce", "argument", "fight", "breakup", "separation", "trust", "jealous"],
            "self_esteem": ["confidence", "worth", "value", "failure", "ugly", "stupid", "incompetent", "loser", "undeserving", "inadequate"],
            "trauma": ["trauma", "abuse", "assault", "accident", "violence", "nightmare", "flashback", "ptsd", "terrify", "horrific"]
        }
        
        # Check user input against keywords
        user_input_lower = user_input.lower()
        
        for issue, keywords in mental_health_keywords.items():
            for keyword in keywords:
                if keyword in user_input_lower:
                    # Increase score based on sentiment - more negative sentiment means higher score
                    increase = 1 + (1 - sentiment["compound"]) * 0.5
                    if sentiment["compound"] < -0.2:  # If negative sentiment
                        increase *= 1.5
                    
                    st.session_state.mental_health_scores[issue] += increase

        # LLM-based assessment for more complex understanding
        if len(user_input) > 15:  # Only for substantial responses
            try:
                assess_prompt = f"""
                Analyze this statement for signs of mental health issues. Give a rating from 0-5 
                (0 = not present, 5 = severe) for each of these categories:
                - Anxiety
                - Depression
                - Stress
                - Loneliness
                - Grief
                - Relationship issues
                - Self-esteem issues
                - Trauma indicators
                
                Return ONLY the numerical ratings in JSON format like this:
                {{
                    "anxiety": X,
                    "depression": X,
                    "stress": X,
                    "loneliness": X,
                    "grief": X,
                    "relationship_issues": X,
                    "self_esteem": X,
                    "trauma": X
                }}
                
                The statement: "{user_input}"
                """
                
                llm_assessment = use_llama3_api(assess_prompt, max_tokens=100)
                
                # Extract the JSON part
                try:
                    json_start = llm_assessment.find('{')
                    json_end = llm_assessment.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = llm_assessment[json_start:json_end]
                        llm_scores = json.loads(json_str)
                        
                        # Update our scores with LLM insights (give them less weight than keyword matches)
                        for issue, score in llm_scores.items():
                            if issue in st.session_state.mental_health_scores:
                                # Add 0.4 points for every point in LLM rating
                                st.session_state.mental_health_scores[issue] += score * 0.4
                except:
                    # If JSON parsing fails, continue without LLM assessment
                    pass
            except:
                # If LLM call fails, continue with just keyword matching
                pass

    # Generate next therapy question and relevant options
    def generate_next_question_with_options(conversation_history, mental_health_scores):
        # Create a prompt for determining the next question with options
        scores_summary = ", ".join([f"{issue}: {score:.1f}" for issue, score in mental_health_scores.items()])
        
        previous_convo = ""
        if conversation_history:
            previous_convo = "\nPrevious conversation: " + " ".join([f"{'User: ' if i%2==1 else 'Therapist: '}{msg}" for i, msg in enumerate(conversation_history)])
        
        prompt = f"""Act as a supportive mental health therapist. You're having a conversation with someone seeking help.
    Current mental health indicators: {scores_summary}
    {previous_convo}

    Based on this information, what's the most important therapeutic question to ask next?
    Also provide 5 likely response options the person might give.

    Format your response as a JSON object like this:
    {{
    "question": "Your therapeutic question here?",
    "options": [
        "Possible response 1",
        "Possible response 2",
        "Possible response 3",
        "Possible response 4",
        "Possible response 5"
    ]
    }}

    Ensure your question is empathetic, supportive, and helps explore the person's feelings or situation further.
    Make the options specific and relevant to potential mental health concerns."""

        try:
            response = use_llama3_api(prompt, max_tokens=500)
            
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
            
            except json.JSONDecodeError:
                # Default question if JSON parsing fails
                return "How are you feeling right now?", [
                    "I'm feeling anxious",
                    "I'm feeling sad",
                    "I'm feeling overwhelmed",
                    "I'm feeling okay",
                    "I don't know how I feel",
                    "Other"
                ]
                
        except Exception as e:
            st.error(f"Error generating question: {str(e)}")
            
        # Fallback if everything else fails
        return "Would you like to tell me more about what's on your mind?", [
            "Yes, I need to talk",
            "I'm not sure where to start",
            "I don't think it will help",
            "I'm feeling too overwhelmed",
            "I'd rather listen to advice",
            "Other"
        ]

    # Generate therapeutic response based on user input
    def generate_therapeutic_response(user_input, conversation_history, mental_health_scores):
        # If using Blenderbot for simple conversation
        if st.session_state.llm_service == "blenderbot":
            tokenizer, model = load_blenderbot_model()
            return use_blenderbot(user_input, tokenizer, model)
        
        # If using LLaMA3 for more sophisticated responses
        scores_summary = ", ".join([f"{issue}: {score:.1f}" for issue, score in mental_health_scores.items()])
        
        # Get previous 5 exchanges to maintain context without making prompt too long
        recent_convo = ""
        if len(conversation_history) > 0:
            # Get up to last 10 messages (5 exchanges)
            recent_messages = conversation_history[-10:] if len(conversation_history) >= 10 else conversation_history
            recent_convo = "\n".join([f"{'User: ' if i%2==1 else 'Therapist: '}{msg}" for i, msg in enumerate(recent_messages)])
        
        # Determine highest scoring mental health issues
        top_issues = sorted(mental_health_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_issues_str = ", ".join([f"{issue}" for issue, score in top_issues if score > 1])
        focus_areas = f"Potential areas of focus: {top_issues_str}" if top_issues_str else "No clear mental health concerns identified yet."
        
        prompt = f"""Act as an empathetic, supportive mental health therapist using person-centered therapy approaches.
    You're having a conversation with someone seeking help.

    Current mental health indicators: {scores_summary}
    {focus_areas}

    Recent conversation:
    {recent_convo}

    User's most recent message: "{user_input}"

    Provide a thoughtful, validating response that:
    1. Shows you understand their feelings
    2. Offers support without judgment
    3. Asks open-ended questions to explore their concerns deeper
    4. Avoids giving simplistic advice or dismissing feelings
    5. Uses techniques like reflective listening and validation

    Keep your response conversational, warm and natural - like a supportive friend would talk.
    Limit your response to 3-4 sentences to maintain engagement."""

        try:
            return use_llama3_api(prompt, max_tokens=250)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "I'm here to listen. Would you like to tell me more about what you're experiencing?"

    # Determine if assessment is complete
    def assessment_complete(mental_health_scores, conversation_length):
        # Check if we have enough information to provide a summary
        # Criteria: At least 5 exchanges and some significant scores
        
        # Count significant issues (score > 2)
        significant_issues = sum(1 for score in mental_health_scores.values() if score > 2)
        
        # Complete assessment if we have:
        # - At least 5 conversation exchanges AND some significant issues identified
        # - OR at least 10 exchanges (regardless of issues identified)
        return (conversation_length >= 10 and significant_issues >= 1) or conversation_length >= 20

    # Generate mental health assessment summary
    def generate_assessment_summary(conversation_history, mental_health_scores):
        # Only include scores that are significant
        significant_scores = {issue: score for issue, score in mental_health_scores.items() if score > 1}
        
        # Sort by score (highest first)
        sorted_scores = sorted(significant_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create a text representation of the scores
        scores_text = ""
        for issue, score in sorted_scores:
            # Convert numeric score to severity level
            severity = "mild"
            if score > 5:
                severity = "moderate"
            if score > 8:
                severity = "significant"
            if score > 12:
                severity = "severe"
                
            formatted_issue = issue.replace("_", " ").title()
            scores_text += f"- {formatted_issue}: {severity} (score: {score:.1f})\n"
        
        if not scores_text:
            scores_text = "- No significant mental health concerns detected\n"
        
        # Selected excerpts from conversation
        sentiment_data = st.session_state.sentiment_analysis
        
        # Get the most negative and most positive statements
        if sentiment_data:
            most_negative = min(sentiment_data, key=lambda x: x["sentiment"]["compound"])
            most_positive = max(sentiment_data, key=lambda x: x["sentiment"]["compound"])
            
            significant_statements = f"""
    Most concerning statement: "{most_negative['text']}"
    Most positive statement: "{most_positive['text']}"
            """
        else:
            significant_statements = "No significant statements analyzed."
        
        # Create prompt for generating the assessment
        prompt = f"""As a mental health professional, create a supportive therapeutic assessment summary 
    based on the following information from a conversation with a client:

    Mental health indicators:
    {scores_text}

    {significant_statements}

    Create a compassionate assessment summary that includes:
    1. The primary mental health concerns identified (if any)
    2. Supportive validation of the person's experiences
    3. General self-care recommendations
    4. When professional help would be recommended
    5. A hopeful message about the possibility of improvement

    Your assessment should be non-judgmental, respectful, and empowering. Focus on the person's 
    strengths as well as challenges. Make it clear this is NOT a clinical diagnosis."""

        try:
            assessment = use_llama3_api(prompt, max_tokens=500)
            
            # Determine overall severity rating
            severity = "Low"
            highest_score = max(mental_health_scores.values()) if mental_health_scores else 0
            if highest_score > 8:
                severity = "High"
            elif highest_score > 4:
                severity = "Moderate"
                
            # Add disclaimer
            assessment += "\n\n[Note: This is an AI-generated assessment for educational purposes only and should not replace professional mental health advice.]"
            
            return assessment, severity
        except Exception as e:
            st.error(f"Error generating assessment: {str(e)}")
            return "Unable to generate a complete assessment at this time. Please consider speaking with a mental health professional for personalized support.", "Unknown"

    # Generate resources based on mental health concerns
    def generate_resources(mental_health_scores):
        # Identify top 3 concerns
        top_concerns = sorted(mental_health_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_concerns = [concern for concern, score in top_concerns if score > 1]
        
        if not top_concerns:
            top_concerns = ["general_wellbeing"]
        
        concerns_text = ", ".join(top_concerns)
        
        prompt = f"""Create a list of helpful resources for someone dealing with these mental health concerns: {concerns_text}.

    Include:
    1. Three self-help techniques they can try immediately
    2. Three types of professionals who might help with these concerns
    3. Two reputable organizations or hotlines that provide support
    4. Two recommended books or workbooks that address these concerns

    Format your response with clear headings and brief explanations. Focus on practical, evidence-based resources."""

        try:
            return use_llama3_api(prompt, max_tokens=400)
        except:
            # Fallback resources if API fails
            return """
    ## Helpful Resources

    ### Self-help Techniques
    - Practice deep breathing exercises (4-7-8 method)
    - Journal about your thoughts and feelings
    - Engage in regular physical activity

    ### Professional Support
    - Licensed therapists or counselors
    - Psychiatrists (for medication evaluation)
    - Support groups for your specific concerns

    ### Support Organizations
    - Crisis Text Line: Text HOME to 741741
    - National Alliance on Mental Health (NAMI): 1-800-950-NAMI (6264)

    ### Recommended Reading
    - "Feeling Good" by David Burns
    - "The Anxiety and Phobia Workbook" by Edmund Bourne

    Remember that seeking help is a sign of strength, not weakness.
    """

    def display_interactive_summary(mental_health_scores, assessment, resources):
        """Display an interactive, visually appealing summary of the therapy session"""
        
        st.markdown("## Your Wellness Summary")
        
        # Create tabs for different sections of the report
        summary_tabs = st.tabs(["Overview", "Insights", "Recommendations", "Resources"])
        
        with summary_tabs[0]:  # Overview tab
            st.markdown("### How You're Feeling")
            
            # Create two columns for layout
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Get top concerns and sort scores for visualization
                sorted_scores = sorted(mental_health_scores.items(), key=lambda x: x[1], reverse=True)
                concerns = [item[0].replace("_", " ").title() for item in sorted_scores if item[1] > 1]
                scores = [item[1] for item in sorted_scores if item[1] > 1]
                
                # If we have concerns to display
                if concerns:
                    # Create color scale
                    colors = []
                    for score in scores:
                        if score > 8:
                            colors.append("#FF4B4B")  # Red for high scores
                        elif score > 4:
                            colors.append("#FFA64B")  # Orange for medium scores
                        else:
                            colors.append("#4B9AFF")  # Blue for low scores
                    
                    # Create horizontal bar chart
                    chart_data = {
                        "concern": concerns,
                        "score": scores
                    }
                    
                    # Use Altair for better visualization if available
                    try:
                        import altair as alt
                        import pandas as pd
                        
                        chart_df = pd.DataFrame(chart_data)
                        
                        chart = alt.Chart(chart_df).mark_bar().encode(
                            x='score',
                            y=alt.Y('concern', sort='-x'),
                            color=alt.Color('score', scale=alt.Scale(domain=[1, 5, 10], range=['#4B9AFF', '#FFA64B', '#FF4B4B'])),
                            tooltip=['concern', 'score']
                        ).properties(
                            title='Mental Health Indicators',
                            height=min(250, len(concerns) * 40)
                        )
                        
                        st.altair_chart(chart, use_container_width=True)
                    except:
                        # Fallback to simple bar chart if Altair isn't available
                        st.bar_chart(chart_data, x="concern", y="score", use_container_width=True)
                else:
                    st.info("No significant mental health concerns were detected.")
            
            with col2:
                # Overall wellness status
                highest_score = max(mental_health_scores.values()) if mental_health_scores else 0
                
                if highest_score > 8:
                    wellness_status = "Needs Attention"
                    status_color = "#FF4B4B"
                elif highest_score > 4:
                    wellness_status = "Moderate Concern"
                    status_color = "#FFA64B"
                else:
                    wellness_status = "Doing Well"
                    status_color = "#4CAF50"
                
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: {status_color}20; 
                border: 1px solid {status_color}; text-align: center; margin-bottom: 20px;">
                    <h3 style="color: {status_color};">Current Status</h3>
                    <h2 style="color: {status_color};">{wellness_status}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Quick mood check
                st.markdown("### How are you feeling right now?")
                mood = st.select_slider(
                    "My current mood is:",
                    options=["Very Low", "Low", "Neutral", "Good", "Great"],
                    value="Neutral"
                )
                
                # Encouragement based on mood
                if mood in ["Very Low", "Low"]:
                    st.markdown("🌱 It's okay to not be okay. Small steps lead to big changes.")
                elif mood == "Neutral":
                    st.markdown("✨ You're doing better than you think. Keep going!")
                else:
                    st.markdown("🌟 That's wonderful! Celebrate your positive moments.")
        
        with summary_tabs[1]:  # Insights tab
            st.markdown("### Understanding Your Experience")
            
            # Extract key points from assessment
            if assessment:
                # Find primary concerns
                import re
                
                # Extract concerns and validation text with improved pattern matching
                concerns_match = re.search(r"Primary Mental Health Concerns:(.*?)(?:Validation|General Self-Care)", assessment, re.DOTALL)
                validation_match = re.search(r"Validation of Your Experiences:(.*?)(?:General Self-Care|When Professional)", assessment, re.DOTALL)
                
                # Clean function to remove asterisks and replace placeholders
                def clean_text(text):
                    # Remove asterisks
                    text = text.replace("**", "")
                    # Replace placeholders
                    text = text.replace("[Client]", "friend").replace("Dear [Client],", "")
                    text = text.replace("[Your Name]", "Your Well-Wisher")
                    return text.strip()
                
                if concerns_match:
                    st.markdown("#### Key Insights")
                    concerns_text = clean_text(concerns_match.group(1).strip())
                    st.info(concerns_text)
                
                if validation_match:
                    st.markdown("#### Reflections")
                    validation_text = clean_text(validation_match.group(1).strip())
                    st.success(validation_text)
                
                # Allow user to see full assessment if desired
                with st.expander("See Complete Analysis"):
                    # Clean the full assessment text before displaying
                    cleaned_assessment = clean_text(assessment)
                    # Replace the formal greeting and signature
                    cleaned_assessment = cleaned_assessment.replace("Dear friend,", "")
                    cleaned_assessment = re.sub(r"Sincerely,.*$", "- Your Well-Wisher", cleaned_assessment)
                    st.write(cleaned_assessment)
            else:
                st.warning("We couldn't generate a detailed assessment. Please speak with a mental health professional for personalized insights.")
        
        with summary_tabs[2]:  # Recommendations tab
            st.markdown("### Suggested Next Steps")
            
            # Create priority recommendations based on top concerns
            top_concerns = sorted(mental_health_scores.items(), key=lambda x: x[1], reverse=True)[:2]
            top_concerns = [concern for concern, score in top_concerns if score > 1]
            
            if not top_concerns:
                top_concerns = ["general_wellbeing"]
            
            # Extract recommendations from resources if available
            recommendations = []
            self_help_match = re.search(r"Self-Help Techniques(.*?)(?:Professionals Who Can Help|Support)", resources, re.DOTALL) if resources else None
            
            if self_help_match:
                techniques = re.findall(r"([\w\s]+):", self_help_match.group(1))
                recommendations = [t.strip() for t in techniques if t.strip()]
            
            # Fallback recommendations if none found
            if not recommendations:
                recommendations = [
                    "Practice deep breathing exercises",
                    "Connect with supportive friends or family",
                    "Engage in physical activity",
                    "Practice mindfulness meditation"
                ]
            
            # Display actionable recommendations
            for i, rec in enumerate(recommendations[:3]):
                col1, col2 = st.columns([1, 20])
                with col1:
                    if st.checkbox("", key=f"rec_{i}", value=False):
                        pass
                with col2:
                    st.markdown(f"**{rec}**")
            
            # Add custom action
            st.markdown("#### Add Your Own Action")
            custom_action = st.text_input("What's one small step you can take today?")
            if custom_action:
                st.success(f"Great! Remember to try: {custom_action}")
            
            # Professional support recommendation based on severity
            highest_score = max(mental_health_scores.values()) if mental_health_scores else 0
            
            st.markdown("#### Professional Support")
            if highest_score > 8:
                st.warning("Based on our conversation, speaking with a mental health professional could be beneficial.")
            elif highest_score > 4:
                st.info("Consider reaching out to a mental health professional if you continue to experience these feelings.")
            else:
                st.success("Continue practicing self-care. Reach out to a professional if you notice your symptoms worsening.")
        
        with summary_tabs[3]:  # Resources tab
            st.markdown("### Helpful Resources")
            
            # Create toggles for different types of resources
            resource_types = ["Crisis Support", "Professional Help", "Self-Help Books", "Mobile Apps", "Support Groups"]
            
            selected_resource = st.radio("What type of resources are you looking for?", resource_types)
            
            # Emergency resources always visible
            if selected_resource == "Crisis Support":
                st.markdown("""
                #### Immediate Support
                - **Crisis Text Line**: Text HOME to 741741 (24/7 support)
                - **National Suicide Prevention Lifeline**: 988 or 1-800-273-8255
                - **Emergency Services**: Call 911 if you're in immediate danger
                """)
            
            elif selected_resource == "Professional Help":
                st.markdown("""
                #### Finding a Therapist
                - **Psychology Today**: Search for therapists in your area
                - **BetterHelp**: Online therapy platform
                - **Your insurance provider**: Many insurance plans cover mental health services
                """)
                
                st.markdown("#### Types of Mental Health Professionals")
                professionals = {
                    "Therapist/Counselor": "Provides talk therapy and emotional support",
                    "Psychiatrist": "Can prescribe medication and provide treatment",
                    "Psychologist": "Specializes in psychological testing and therapy"
                }
                
                for prof, desc in professionals.items():
                    st.markdown(f"**{prof}**: {desc}")
            
            elif selected_resource == "Self-Help Books":
                # Extract book recommendations if available
                books = []
                books_match = re.search(r"Recommended (Books|Reading)(.*?)(?:\[Note|\Z)", resources, re.DOTALL) if resources else None
                
                if books_match:
                    book_text = books_match.group(2)
                    books = re.findall(r'"([^"]+)"', book_text)
                
                # Fallback books if none found
                if not books:
                    books = [
                        "Feeling Good by David Burns",
                        "The Anxiety and Phobia Workbook by Edmund Bourne",
                        "Man's Search for Meaning by Viktor Frankl"
                    ]
                
                for book in books:
                    st.markdown(f"- **{book}**")
            
            elif selected_resource == "Mobile Apps":
                st.markdown("""
                #### Helpful Mobile Apps
                - **Headspace**: Guided meditation and mindfulness exercises
                - **Calm**: Sleep, meditation and relaxation aid
                - **Woebot**: AI chatbot for mental health support
                - **Daylio**: Mood tracking journal
                - **Breathe2Relax**: Guided breathing exercises
                """)
            
            elif selected_resource == "Support Groups":
                st.markdown("""
                #### Finding Support Groups
                - **NAMI**: National Alliance on Mental Illness offers support groups
                - **Mental Health America**: Provides peer support group resources
                - **Support Group Central**: Online support groups for various needs
                
                Remember that connecting with others who understand your experience can be incredibly healing.
                """)
            
            # Option to download resources as PDF
            st.markdown("### Save These Resources")
            if st.button("Prepare Resources PDF"):
                st.success("Your personalized resource list has been prepared!")
                st.markdown("""
                **Note:** In a full implementation, this would generate a downloadable PDF with 
                all relevant resources customized to the user's needs.
                """)
        
        # Final encouragement message
        st.markdown("---")
        st.markdown("""
        ### Remember
        Your mental health journey is unique. Small steps forward still move you in the right direction.
        Each day is a new opportunity to prioritize your wellbeing.
        """)
        
        # Disclaimer
        st.caption("This summary is for educational purposes only and should not replace professional mental health advice.")
        
        # Options for next steps
        st.markdown("### What would you like to do next?")
        next_steps = st.columns(3)
        
        with next_steps[0]:
            if st.button("Continue Talking"):
                return "continue"
        
        with next_steps[1]:
            if st.button("Start New Session"):
                return "new"
        
        with next_steps[2]:
            if st.button("End Session"):
                return "end"
        
        return None

    st.title("AI Therapy Assistant")
    st.markdown("_This is a prototype for educational purposes only and should not be used as a replacement for professional mental health services._")
    
    def submit_initial_response():
        if st.session_state.initial_response:
            # Record the conversation - AI message first, then user
            st.session_state.conversation.append(initial_greeting)
            st.session_state.conversation.append(st.session_state.initial_response)
            
            # Update mental health scores based on initial response
            update_mental_health_scores(st.session_state.initial_response)
            
            # Generate first therapeutic response
            first_response = generate_therapeutic_response(
                st.session_state.initial_response,
                st.session_state.conversation,
                st.session_state.mental_health_scores
            )
            
            # Add to conversation
            st.session_state.conversation.append(first_response)
            
            # Change state to therapy
            st.session_state.therapy_state = "therapy"
            
            # Prepare next question with options
            question, options = generate_next_question_with_options(
                st.session_state.conversation,
                st.session_state.mental_health_scores
            )
            
            st.session_state.current_options = options
            st.session_state.user_submitted = False  # Change to false since we don't want auto response

    def handle_custom_submit():
        if st.session_state.custom_input:
            # Record the response in conversation
            st.session_state.conversation.append(st.session_state.custom_input)
            
            # Update mental health scores
            update_mental_health_scores(st.session_state.custom_input)
            
            # Reset other_selected flag
            st.session_state.other_selected = False
            
            # Set user_submitted flag
            st.session_state.user_submitted = True
            
            # Clear input - avoids duplicate submissions
            st.session_state.custom_input = ""

    # Side panel for controls
    with st.sidebar:
        st.header("Controls")
        
        # LLM Service Selection
        st.subheader("AI Model")
        llm_option = st.radio(
            "Select AI Model", 
            ["LLaMA3-70B (Advanced)", "Blenderbot (Simple Conversation)"],
            index=0 if st.session_state.llm_service == "llama3" else 1
        )
        
        # Update LLM service based on selection
        if (llm_option == "LLaMA3-70B (Advanced)" and st.session_state.llm_service != "llama3") or \
        (llm_option == "Blenderbot (Simple Conversation)" and st.session_state.llm_service != "blenderbot"):
            # Store the new service selection
            st.session_state.llm_service = "llama3" if llm_option == "LLaMA3-70B (Advanced)" else "blenderbot"
            
            # Reset conversation state
            st.session_state.conversation = []
            st.session_state.therapy_state = "initial"
            st.session_state.mental_health_scores = {
                "anxiety": 0,
                "depression": 0,
                "stress": 0, 
                "loneliness": 0,
                "grief": 0,
                "relationship_issues": 0,
                "self_esteem": 0,
                "trauma": 0
            }
            st.session_state.assessment_progress = 0
            st.session_state.current_question = None
            st.session_state.current_options = None
            st.session_state.user_submitted = False
            st.session_state.severity_rating = None
            st.session_state.sentiment_analysis = []
            st.session_state.other_selected = False
            
            # Show notification
            st.success(f"Switched to {llm_option}. Starting new conversation.")
            st.rerun()
        
        # Display mental health scores
        if st.session_state.mental_health_scores:
            st.subheader("Mental Health Indicators")
            for issue, score in sorted(st.session_state.mental_health_scores.items(), key=lambda x: x[1], reverse=True):
                # Only show scores with some significance
                if score > 0.5:
                    # Format the issue name for display
                    display_name = issue.replace("_", " ").title()
                    # Create color gradient based on score
                    color_intensity = min(score / 15, 1.0)  # Max at 15
                    color = f"rgba(255, {int(255*(1-color_intensity))}, {int(255*(1-color_intensity))}, 0.8)"
                    st.markdown(
                        f"""<div style="background-color: {color}; padding: 5px; border-radius: 5px;">
                        {display_name}: {score:.1f}</div>""", 
                        unsafe_allow_html=True
                    )
        
        # Settings section
        st.subheader("Settings")
        delay_option = st.checkbox("Simulate therapist typing delay", value=st.session_state.response_delay)
        if delay_option != st.session_state.response_delay:
            st.session_state.response_delay = delay_option
        
        # Reset conversation button
        st.subheader("Session")
        if st.button("Start New Conversation"):
            st.session_state.conversation = []
            st.session_state.therapy_state = "initial"
            st.session_state.mental_health_scores = {
                "anxiety": 0,
                "depression": 0,
                "stress": 0,
                "loneliness": 0,
                "grief": 0,
                "relationship_issues": 0,
                "self_esteem": 0,
                "trauma": 0
            }
            st.session_state.assessment_progress = 0
            st.session_state.current_question = None
            st.session_state.current_options = None
            st.session_state.user_submitted = False
            st.session_state.severity_rating = None
            st.session_state.sentiment_analysis = []
            st.session_state.other_selected = False
            st.rerun()
    
    # Main chat interface
    st.header("Therapeutic Conversation")
    
    # Display selected model
    st.caption(f"Using {'LLaMA3-70B' if st.session_state.llm_service == 'llama3' else 'Blenderbot'} for conversation")
    
    # Chat container for better styling
    chat_container = st.container()
    
    # Display conversation history
    with chat_container:
        if st.session_state.conversation:
            for i, message in enumerate(st.session_state.conversation):
                if i % 2 == 1:  # User messages (odd indices)
                    with st.chat_message("user"):
                        st.write(message)
                else:  # AI messages (even indices)
                    with st.chat_message("assistant", avatar="🧠"):
                        st.write(message)
        else:
            st.write("No conversation history yet.")
    
    # Check for end session state first - this is a new state we'll add
    if hasattr(st.session_state, 'therapy_state') and st.session_state.therapy_state == "ended":
        # When the session is ended, we don't need to display any additional UI elements
        # The thank you message should already be in the conversation history
        pass
    
    # Initial greeting - only show if conversation is empty
    elif st.session_state.therapy_state == "initial":
        initial_greeting = "Hello, I'm here to provide a safe space for you to talk. How are you feeling today?"
        
        with chat_container:
            with st.chat_message("assistant", avatar="🧠"):
                st.write(initial_greeting)
        
        # Add an input field for initial response
        with st.form(key="initial_response_form"):
            st.text_area("Share how you're feeling:", key="initial_response", height=100)
            st.form_submit_button("Send", on_click=submit_initial_response)
    
    # Therapeutic conversation phase
    elif st.session_state.therapy_state == "therapy":
        # Check if we should provide an assessment
        if assessment_complete(st.session_state.mental_health_scores, len(st.session_state.conversation)):
            with st.spinner("Preparing assessment..."):
                assessment, severity = generate_assessment_summary(
                    st.session_state.conversation,
                    st.session_state.mental_health_scores
                )
                
                # Generate helpful resources
                resources = generate_resources(st.session_state.mental_health_scores)
                
                # Store results
                st.session_state.assessment = assessment
                st.session_state.resources = resources
                st.session_state.severity_rating = severity
                st.session_state.therapy_state = "summary"
                
                # Add transition message to conversation
                transition_message = "I've had a chance to reflect on our conversation. Would you like to see a summary of what I'm hearing from you, along with some resources that might be helpful?"
                st.session_state.conversation.append(transition_message)
                
                # Rerun to show the new state
                st.rerun()
        
        # Process user input and generate response
        if st.session_state.user_submitted:
            # Generate therapeutic response
            last_user_message = st.session_state.conversation[-1]
            
            with st.spinner("Thinking..."):
                # Apply optional delay to simulate typing
                if st.session_state.response_delay:
                    time.sleep(1.5)
                
                # Generate response
                response = generate_therapeutic_response(
                    last_user_message,
                    st.session_state.conversation,
                    st.session_state.mental_health_scores
                )
                
                # Add to conversation
                st.session_state.conversation.append(response)
                
                # Prepare next question with options
                question, options = generate_next_question_with_options(
                    st.session_state.conversation,
                    st.session_state.mental_health_scores
                )
                
                st.session_state.current_options = options
                
                # Reset user_submitted flag
                st.session_state.user_submitted = False
                
                # Just rerun to refresh the page with the new conversation state
                st.rerun()
        
        # Show free-form input or options for user
        if not st.session_state.user_submitted and not st.session_state.other_selected:
            # First, show options if we have them
            if st.session_state.current_options:
                options_container = st.container()
                with options_container:
                    st.radio(
                        "Quick responses:",
                        st.session_state.current_options,
                        key="selected_option"
                    )
                    
                    cols = st.columns([1, 1])
                    with cols[0]:
                        if st.button("Send Quick Response"):
                            if "selected_option" in st.session_state and st.session_state.selected_option:
                                if st.session_state.selected_option == "Other":
                                    st.session_state.other_selected = True
                                else:
                                    # Record the response in conversation
                                    st.session_state.conversation.append(st.session_state.selected_option)
                                    
                                    # Update mental health scores
                                    update_mental_health_scores(st.session_state.selected_option)
                                    
                                    st.session_state.user_submitted = True
                                st.rerun()
                    
                    with cols[1]:
                        if st.button("I'd prefer to type my response"):
                            st.session_state.other_selected = True
                            st.rerun()
            
            # Show free-form text input if "Other" is selected or user prefers typing
            if st.session_state.other_selected:
                with st.form(key="custom_response_form"):
                    st.text_area("Your response:", key="custom_input", height=100)
                    st.form_submit_button("Send", on_click=handle_custom_submit)
    
    # Summary and resources phase
    elif st.session_state.therapy_state == "summary":
        # Define summary choice submission handler
        def submit_summary_choice():
            if st.session_state.summary_choice:
                # Add user response to conversation
                st.session_state.conversation.append(st.session_state.summary_choice)
                
                # If user wants to see summary, show it
                if "yes" in st.session_state.summary_choice.lower():
                    # Set flag to display interactive summary instead of text summary
                    st.session_state.show_interactive_summary = True
                else:
                    # Continue conversation
                    st.session_state.conversation.append("That's completely fine. We can continue our conversation. What would you like to talk about next?")
                    st.session_state.therapy_state = "therapy"
        
        # Add a respond to transition message form if not already responded
        if len(st.session_state.conversation) % 2 == 1:  # Odd number means waiting for user response
            with st.form(key="summary_choice_form"):
                st.radio(
                    "Would you like to see a summary and helpful resources?",
                    ["Yes, I'd like to see the summary", "No, I'd prefer to continue talking"],
                    key="summary_choice"
                )
                st.form_submit_button("Send", on_click=submit_summary_choice)
        
        # If user has seen summary and we're waiting for their next message
        elif len(st.session_state.conversation) % 2 == 0 and len(st.session_state.conversation) >= 2:
            # Display interactive summary if flag is set
            if hasattr(st.session_state, 'show_interactive_summary') and st.session_state.show_interactive_summary:
                # Use the display_interactive_summary function directly here
                # instead of adding assessment to conversation first
                next_action = display_interactive_summary(
                    st.session_state.mental_health_scores,
                    st.session_state.assessment,
                    st.session_state.resources
                )
                
                # Handle the return value from the interactive summary
                if next_action == "continue":
                    # Don't use rerun here - update session state only
                    st.session_state.therapy_state = "therapy"
                    st.session_state.conversation.append("Let's continue our conversation. What's on your mind?")
                    st.session_state.show_interactive_summary = False
                elif next_action == "new":
                    # Reset for new conversation
                    st.session_state.conversation = []
                    st.session_state.therapy_state = "initial"
                    st.session_state.mental_health_scores = {
                        "anxiety": 0,
                        "depression": 0,
                        "stress": 0,
                        "loneliness": 0,
                        "grief": 0,
                        "relationship_issues": 0,
                        "self_esteem": 0,
                        "trauma": 0
                    }
                    st.session_state.assessment_progress = 0
                    st.session_state.current_question = None
                    st.session_state.current_options = None
                    st.session_state.user_submitted = False
                    st.session_state.severity_rating = None
                    st.session_state.sentiment_analysis = []
                    st.session_state.other_selected = False
                    st.session_state.show_interactive_summary = False
                elif next_action == "end":
                    # Add ending message and set state to "ended"
                    st.session_state.conversation.append("Thank you for talking with me today. Remember that this is just a simulation for educational purposes. If you're experiencing mental health challenges, please consider reaching out to a professional. Take care of yourself.")
                    st.session_state.therapy_state = "ended"  # New state for ended sessions
                    st.session_state.show_interactive_summary = False
                    st.rerun()  # Rerun to refresh UI
            else:
                # This block is for after the interactive summary has been shown
                # or if the user chose to see text summary instead
                
                # First check if we need to add the text assessment to conversation
                # (only do this if interactive summary has been shown and closed)
                if hasattr(st.session_state, 'show_interactive_summary') and not st.session_state.show_interactive_summary and not any(st.session_state.assessment in msg for msg in st.session_state.conversation):
                    st.session_state.conversation.append(st.session_state.assessment + "\n\n" + st.session_state.resources)
                
                # After showing summary, offer options
                final_options = [
                    "I'd like to continue our conversation",
                    "I found this helpful, thank you",
                    "I'd like to start a new conversation",
                    "I'd like to learn more about specific resources",
                    "Show me an interactive summary"  # Added option for interactive summary
                ]
                
                def handle_final_choice():
                    if st.session_state.final_choice:
                        # Add user choice to conversation
                        st.session_state.conversation.append(st.session_state.final_choice)
                        
                        if "continue" in st.session_state.final_choice.lower():
                            # Return to therapy state
                            st.session_state.therapy_state = "therapy"
                            st.session_state.conversation.append("I'm here to continue our conversation. What's on your mind?")
                        elif "new" in st.session_state.final_choice.lower():
                            # Reset for new conversation
                            st.session_state.conversation = []
                            st.session_state.therapy_state = "initial"
                            st.session_state.mental_health_scores = {
                                "anxiety": 0,
                                "depression": 0,
                                "stress": 0,
                                "loneliness": 0,
                                "grief": 0,
                                "relationship_issues": 0,
                                "self_esteem": 0,
                                "trauma": 0
                            }
                            st.session_state.assessment_progress = 0
                            st.session_state.current_question = None
                            st.session_state.current_options = None
                            st.session_state.user_submitted = False
                            st.session_state.severity_rating = None
                            st.session_state.sentiment_analysis = []
                            st.session_state.other_selected = False
                            st.rerun()

                        elif "resources" in st.session_state.final_choice.lower():
                            # Generate more specific resources
                            with st.spinner("Finding more specific resources..."):
                                detailed_resources = use_llama3_api(
                                    "Provide a detailed list of mental health resources including specific apps, websites, hotlines, and books. Include resources for both immediate crisis and long-term support.",
                                    max_tokens=600
                                )
                                st.session_state.conversation.append("Here are some more detailed resources that might be helpful:\n\n" + detailed_resources)
                        elif "interactive summary" in st.session_state.final_choice.lower():
                            # Show interactive summary
                            st.session_state.show_interactive_summary = True
                        elif 'end' in st.session_state.final_choice.lower() or "thank" in st.session_state.final_choice.lower():
                            # Add ending message and set state to "ended"
                            st.session_state.conversation.append("Thank you for talking with me today. Remember that this is just a simulation for educational purposes. If you're experiencing mental health challenges, please consider reaching out to a professional. Take care of yourself.")
                            st.session_state.therapy_state = "ended"  # New state for ended sessions
                            st.rerun()  # Rerun to refresh UI
                
                with st.form(key="final_choice_form"):
                    st.radio(
                        "What would you like to do next?",
                        final_options,
                        key="final_choice"
                    )
                    st.form_submit_button("Send", on_click=handle_final_choice)
                    
                    # Add End Session button
                    if st.form_submit_button("End Session"):
                        # Add ending message and set state to "ended"
                        st.session_state.conversation.append("Thank you for talking with me today. Remember that this is just a simulation for educational purposes. If you're experiencing mental health challenges, please consider reaching out to a professional. Take care of yourself.")
                        st.session_state.therapy_state = "ended"  # New state for ended sessions
                        st.rerun()  # Rerun to refresh UI
    
    # Footer with disclaimer
    st.markdown("---")
    st.caption("**IMPORTANT DISCLAIMER:** This is an educational prototype only and should not be used for actual mental health support. The AI models used have limitations and may not provide accurate or appropriate responses. If you're experiencing mental health issues, please contact a qualified healthcare professional or a crisis service such as the National Suicide Prevention Lifeline at 988 or 1-800-273-8255.")

    # Back button
    if st.button("Back to Main Menu"):
        st.session_state.page = "main"
        st.rerun()

# Run the application
if __name__ == "__main__":
    run_mental_health()