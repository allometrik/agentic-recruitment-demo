from typing import Literal, Tuple, Dict, Optional
import os
import time
import json
import requests
import PyPDF2
from datetime import datetime, timedelta
import pytz
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.utils import formataddr

import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.email import EmailTools
from agno.tools import Toolkit
from agno.utils.log import log_info, logger as agno_logger
from phi.tools.zoom import ZoomTool
from phi.utils.log import logger
from streamlit_pdf_viewer import pdf_viewer



class CustomZoomTool(ZoomTool):
    def __init__(self, *, account_id: Optional[str] = None, client_id: Optional[str] = None, client_secret: Optional[str] = None, name: str = "zoom_tool"):
        super().__init__(account_id=account_id, client_id=client_id, client_secret=client_secret, name=name)
        self.token_url = "https://zoom.us/oauth/token"
        self.access_token = None
        self.token_expires_at = 0

    def get_access_token(self) -> str:
        if self.access_token and time.time() < self.token_expires_at:
            return str(self.access_token)
            
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {"grant_type": "account_credentials", "account_id": self.account_id}

        try:
            response = requests.post(self.token_url, headers=headers, data=data, auth=(self.client_id, self.client_secret))
            response.raise_for_status()

            token_info = response.json()
            self.access_token = token_info["access_token"]
            expires_in = token_info["expires_in"]
            self.token_expires_at = time.time() + expires_in - 60

            self._set_parent_token(str(self.access_token))
            return str(self.access_token)

        except requests.RequestException as e:
            logger.error(f"Error fetching access token: {e}")
            return ""

    def _set_parent_token(self, token: str) -> None:
        """Helper method to set the token in the parent ZoomTool class"""
        if token:
            self._ZoomTool__access_token = token


class CustomEmailTools(Toolkit):
    """Custom EmailTools with proper UTF-8 encoding and character cleaning"""
    def __init__(
        self,
        receiver_email: Optional[str] = None,
        sender_name: Optional[str] = None,
        sender_email: Optional[str] = None,
        sender_passkey: Optional[str] = None,
        **kwargs,
    ):
        self.receiver_email: Optional[str] = receiver_email
        self.sender_name: Optional[str] = sender_name
        self.sender_email: Optional[str] = sender_email
        self.sender_passkey: Optional[str] = sender_passkey

        tools = [self.email_user]
        super().__init__(name="email_tools", tools=tools, **kwargs)

    def _clean_text(self, text: str) -> str:
        """Remove non-breaking spaces and other problematic Unicode characters."""
        if not text:
            return text
        # Replace various types of spaces and problematic characters
        cleaned = text.replace('\xa0', ' ').replace('\u00a0', ' ').replace('\u202f', ' ')
        # Remove any other non-ASCII characters for safety
        cleaned = ''.join(char if ord(char) < 128 else ' ' for char in cleaned)
        return cleaned.strip()

    def email_user(self, subject: str, body: str, **kwargs) -> str:
        """Emails the user with the given subject and body with proper UTF-8 encoding.

        :param subject: The subject of the email.
        :param body: The body of the email.
        :return: "success" if the email was sent successfully, "error: [error message]" otherwise.
        """
        if not self.receiver_email:
            return "error: No receiver email provided"
        if not self.sender_name:
            return "error: No sender name provided"
        if not self.sender_email:
            return "error: No sender email provided"
        if not self.sender_passkey:
            return "error: No sender passkey provided"

        # Clean ALL text fields - convert all non-ASCII to ASCII
        subject_clean = self._clean_text(subject)
        body_clean = self._clean_text(body)
        sender_name_clean = self._clean_text(self.sender_name)
        sender_email_clean = self._clean_text(self.sender_email).strip()
        receiver_email_clean = self._clean_text(self.receiver_email).strip()
        # Clean passkey too (remove all whitespace)
        passkey_clean = ''.join(self.sender_passkey.split())

        # Use MIMEText for better encoding control
        msg = MIMEText(body_clean, 'plain', 'utf-8')
        msg['Subject'] = subject_clean
        msg['From'] = formataddr((sender_name_clean, sender_email_clean))
        msg['To'] = receiver_email_clean

        log_info(f"Sending Email to {receiver_email_clean}")
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(sender_email_clean, passkey_clean)
                smtp.send_message(msg)
                log_info(f"Email sent successfully to {receiver_email_clean}")
        except Exception as e:
            agno_logger.error(f"Error sending email: {e}")
            return f"error: {e}"
        return "email sent successfully"


# Configuration loaded from Streamlit secrets
# These should be set in Streamlit Cloud under Settings > Secrets
# or locally in .streamlit/secrets.toml

# Role display names mapping
ROLE_DISPLAY_NAMES: Dict[str, str] = {
    "AI/ML Engineer": "ai_ml_engineer",
    "Frontend Engineer": "frontend_engineer",
    "Backend Engineer": "backend_engineer",
    "SDR (Sales Development Representative)": "sdr",
    "HR Manager": "hr_manager"
}

# Role requirements as a constant dictionary
ROLE_REQUIREMENTS: Dict[str, str] = {
    "ai_ml_engineer": """
        Required Skills:
        - Python, PyTorch/TensorFlow
        - Machine Learning algorithms and frameworks
        - Deep Learning and Neural Networks
        - Data preprocessing and analysis
        - MLOps and model deployment
        - RAG, LLM, Finetuning and Prompt Engineering
    """,

    "frontend_engineer": """
        Required Skills:
        - React/Vue.js/Angular
        - HTML5, CSS3, JavaScript/TypeScript
        - Responsive design
        - State management
        - Frontend testing
    """,

    "backend_engineer": """
        Required Skills:
        - Python/Java/Node.js
        - REST APIs
        - Database design and management
        - System architecture
        - Cloud services (AWS/GCP/Azure)
        - Kubernetes, Docker, CI/CD
    """,

    "sdr": """
        Required Skills:
        - Sales prospecting and lead generation
        - CRM proficiency (Salesforce, HubSpot, etc.)
        - Cold calling and email outreach
        - Communication and negotiation skills
        - B2B sales experience
        - Sales pipeline management
        - Meeting scheduling and follow-ups
        - Performance metrics tracking (KPIs, conversion rates)
    """,

    "hr_manager": """
        Required Skills:
        - Talent acquisition and recruitment
        - HR policies and labor law knowledge
        - Employee relations and conflict resolution
        - Performance management systems
        - HRIS and ATS platforms
        - Compensation and benefits administration
        - Onboarding and training programs
        - Data-driven decision making
    """
}


def init_session_state() -> None:
    """Initialize session state variables with values from Streamlit secrets."""
    # Load secrets from Streamlit configuration
    try:
        defaults = {
            'candidate_email': "",
            'openai_api_key': st.secrets["OPENAI_API_KEY"],
            'resume_text': "",
            'analysis_complete': False,
            'is_selected': False,
            'zoom_account_id': st.secrets["ZOOM_ACCOUNT_ID"],
            'zoom_client_id': st.secrets["ZOOM_CLIENT_ID"],
            'zoom_client_secret': st.secrets["ZOOM_CLIENT_SECRET"],
            'email_sender': st.secrets["EMAIL_SENDER"],
            'email_passkey': st.secrets["EMAIL_PASSWORD"],
            'company_name': st.secrets.get("COMPANY_NAME", "Allometrik"),
            'current_pdf': None
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    except KeyError as e:
        st.error(f"Missing required secret: {e}")
        st.info("""
        Please configure the following secrets in Streamlit Cloud:
        - OPENAI_API_KEY
        - ZOOM_ACCOUNT_ID
        - ZOOM_CLIENT_ID
        - ZOOM_CLIENT_SECRET
        - EMAIL_SENDER
        - EMAIL_PASSWORD
        - COMPANY_NAME (optional, defaults to 'Allometrik')
        
        Go to: App Settings > Secrets
        """)
        st.stop()


def create_resume_analyzer() -> Agent:
    """Creates and returns a resume analysis agent."""
    if not st.session_state.openai_api_key:
        st.error("Please enter your OpenAI API key first.")
        return None

    return Agent(
        model=OpenAIChat(
            id="gpt-4o",
            api_key=st.session_state.openai_api_key
        ),
        description="You are an expert technical recruiter who analyzes resumes.",
        instructions=[
            "Analyze the resume against the provided job requirements",
            "Be lenient with AI/ML candidates who show strong potential",
            "Consider project experience as valid experience",
            "Value hands-on experience with key technologies",
            "Return a JSON response with selection decision and feedback"
        ],
        markdown=True
    )

def create_email_agent() -> Agent:
    return Agent(
        model=OpenAIChat(
            id="gpt-4o",
            api_key=st.session_state.openai_api_key
        ),
        tools=[CustomEmailTools(
            receiver_email=st.session_state.candidate_email,
            sender_email=st.session_state.email_sender,
            sender_name=st.session_state.company_name,
            sender_passkey=st.session_state.email_passkey
        )],
        description="You are a professional recruitment coordinator handling email communications.",
        instructions=[
            "Draft and send professional recruitment emails with proper capitalization",
            "Use proper grammar and capitalize sentences, names, and titles appropriately",
            "Do NOT use any markdown formatting (no **, __, *, etc.) in email body - write plain text only",
            "Maintain a friendly yet professional tone",
            "Always end emails with exactly: 'Best,\nThe Allometrik Recruiting Team'",
            "Never include the sender's or receiver's name in the signature",
            f"The name of the company is '{st.session_state.company_name}'"
        ],
        markdown=True
    )


def create_scheduler_agent() -> Agent:
    zoom_tools = CustomZoomTool(
        account_id=st.session_state.zoom_account_id,
        client_id=st.session_state.zoom_client_id,
        client_secret=st.session_state.zoom_client_secret
    )

    return Agent(
        name="Interview Scheduler",
        model=OpenAIChat(
            id="gpt-4o",
            api_key=st.session_state.openai_api_key
        ),
        tools=[zoom_tools],
        description="You are an interview scheduling coordinator.",
        instructions=[
            "You are an expert at scheduling technical interviews.",
            "Schedule interviews during business hours (9 AM - 5 PM IST)",
            "Create meetings with proper titles and descriptions",
            "Ensure all meeting details are included in responses",
            "Use ISO 8601 format for dates",
            "Handle scheduling errors gracefully",
            "When presenting meeting details, refer to the platform as Microsoft Teams"
        ],
        markdown=True
    )


def extract_text_from_pdf(pdf_file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting PDF text: {str(e)}")
        return ""


def extract_email_from_resume(resume_text: str) -> Optional[str]:
    """Extract candidate's email address from resume text using LLM."""
    try:
        agent = Agent(
            model=OpenAIChat(
                id="gpt-4o-mini",  # Using mini for faster/cheaper extraction
                api_key=st.session_state.openai_api_key
            ),
            description="You are an email extraction specialist.",
            instructions=[
                "Extract the candidate's email address from the resume text",
                "Return ONLY the email address, nothing else",
                "If no email is found, return 'NOT_FOUND'",
                "Do not add any explanation or formatting"
            ],
            markdown=False
        )
        
        response = agent.run(f"Extract the email address from this resume:\n\n{resume_text}")
        
        # Get the assistant's response
        assistant_message = next((msg.content for msg in response.messages if msg.role == 'assistant'), None)
        
        if assistant_message and assistant_message.strip() != "NOT_FOUND":
            email = assistant_message.strip()
            # Basic validation - check if it looks like an email
            if '@' in email and '.' in email:
                return email
        
        return None
        
    except Exception as e:
        log_info(f"Error extracting email: {str(e)}")
        return None


def analyze_resume(
    resume_text: str,
    role: Literal["ai_ml_engineer", "frontend_engineer", "backend_engineer", "sdr", "hr_manager"],
    analyzer: Agent
) -> Tuple[bool, str]:
    try:
        response = analyzer.run(
            f"""Please analyze this resume against the following requirements and provide your response in valid JSON format:
            Role Requirements:
            {ROLE_REQUIREMENTS[role]}
            Resume Text:
            {resume_text}
            Your response must be a valid JSON object like this:
            {{
                "selected": true/false,
                "feedback": "Detailed feedback explaining the decision",
                "matching_skills": ["skill1", "skill2"],
                "missing_skills": ["skill3", "skill4"],
                "experience_level": "junior/mid/senior"
            }}
            Evaluation criteria:
            1. Match at least 70% of required skills
            2. Consider both theoretical knowledge and practical experience
            3. Value project experience and real-world applications
            4. Consider transferable skills from similar technologies
            5. Look for evidence of continuous learning and adaptability
            Important: Return ONLY the JSON object without any markdown formatting or backticks.
            """
        )

        assistant_message = next((msg.content for msg in response.messages if msg.role == 'assistant'), None)
        if not assistant_message:
            raise ValueError("No assistant message found in response.")

        result = json.loads(assistant_message.strip())
        if not isinstance(result, dict) or not all(k in result for k in ["selected", "feedback"]):
            raise ValueError("Invalid response format")

        return result["selected"], result["feedback"]

    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Error processing response: {str(e)}")
        return False, f"Error analyzing resume: {str(e)}"


def send_selection_email(email_agent: Agent, to_email: str, role: str, role_display: str) -> None:
    email_agent.run(
        f"""
        Send an email to {to_email} regarding their application for the {role_display} position at Allometrik.
        The email should:
        1. Congratulate them on being selected for an interview
        2. Make it clear they are moving to the interview stage of the recruitment process
        3. Use proper capitalization and professional formatting
        4. Explain the next steps in the process
        5. Mention that they will receive interview details shortly
        6. Do NOT use markdown formatting in the email body
        """
    )


def send_rejection_email(email_agent: Agent, to_email: str, role: str, role_display: str, feedback: str) -> None:
    """
    Send a rejection email with constructive feedback.
    """
    email_agent.run(
        f"""
        Send an email to {to_email} regarding their application for the {role_display} position.
        Use this specific style:
        1. Use proper capitalization and professional formatting
        2. Be empathetic and human
        3. Mention specific feedback from: {feedback}
        4. Encourage them to upskill and try again
        5. Suggest some learning resources based on missing skills
        6. Do NOT use markdown formatting in the email body
        7. End the email with exactly:
           Best,
           The Allometrik Recruiting Team
        
        Do not include any names in the signature.
        The tone should be professional yet empathetic.
        """
    )


def generate_interview_questions(resume_text: str, role: str, role_display: str) -> list:
    """Generate 3 killer interview questions based on the resume and role."""
    try:
        agent = Agent(
            model=OpenAIChat(
                id="gpt-4o",
                api_key=st.session_state.openai_api_key
            ),
            description="You are an expert technical interviewer.",
            instructions=[
                "Generate exactly 3 insightful technical interview questions",
                "Questions should be based on the candidate's experience and the role requirements",
                "Make questions challenging but fair",
                "Focus on practical scenarios and problem-solving",
                "Each question should probe depth of knowledge in key areas"
            ],
            markdown=False
        )
        
        response = agent.run(
            f"""Based on this resume and the {role_display} position, generate 3 killer interview questions.

Resume:
{resume_text}

Role Requirements:
{ROLE_REQUIREMENTS[role]}

Return exactly 3 questions, numbered 1-3, each on a new line. Make them specific to the candidate's experience."""
        )
        
        assistant_message = next((msg.content for msg in response.messages if msg.role == 'assistant'), None)
        
        if assistant_message:
            # Parse the questions
            lines = [line.strip() for line in assistant_message.strip().split('\n') if line.strip()]
            questions = []
            for line in lines:
                # Remove numbering if present (handles "1.", "1)", "1 -", etc.)
                if len(line) > 0 and line[0].isdigit():
                    # Find where the actual question starts
                    for i, char in enumerate(line):
                        if char.isalpha() or char == '"':
                            questions.append(line[i:].strip())
                            break
                else:
                    questions.append(line)
            return questions[:3]  # Return only first 3
        
        return []
        
    except Exception as e:
        log_info(f"Error generating interview questions: {str(e)}")
        return []


def schedule_interview(scheduler: Agent, candidate_email: str, email_agent: Agent, role: str, role_display: str) -> None:
    """
    Schedule interviews during business hours (9 AM - 5 PM IST).
    """
    try:
        # Get current time in IST
        ist_tz = pytz.timezone('Asia/Kolkata')
        current_time_ist = datetime.now(ist_tz)

        tomorrow_ist = current_time_ist + timedelta(days=1)
        interview_time = tomorrow_ist.replace(hour=11, minute=0, second=0, microsecond=0)
        formatted_time = interview_time.strftime('%Y-%m-%dT%H:%M:%S')

        meeting_response = scheduler.run(
            f"""Schedule a 60-minute technical interview with these specifications:
            - Title: '{role_display} Technical Interview'
            - Date: {formatted_time}
            - Timezone: IST (India Standard Time)
            - Attendee: {candidate_email}
            
            Important Notes:
            - The meeting must be between 9 AM - 5 PM IST
            - Use IST (UTC+5:30) timezone for all communications
            - Include timezone information in the meeting details
            - Return the meeting details in a structured format with date, time, duration, and platform
            """
        )

        email_agent.run(
            f"""Send an interview confirmation email with these details:
            - Role: {role_display} position
            - Meeting Details: {meeting_response}
            
            Important instructions:
            - Do NOT use markdown formatting (no **, __, etc.) in the email body - use plain text only
            - Mention that the meeting platform is Microsoft Teams
            - Clearly specify that the time is in IST (India Standard Time)
            - Include these placeholder texts for meeting access:
              * "Join Teams Meeting: [Link will be provided here]"
              * "Meeting ID: [Meeting ID will be provided here]"
              * "Passcode: [Passcode will be provided here]"
            - Ask the candidate to join 5 minutes early to avoid any last-minute technical issues
            - Encourage them to be confident and well-prepared as this is an opportunity to showcase their skills and knowledge
            - Format the meeting details in a clean, readable way without markdown symbols
            - Do NOT include any timezone conversion links
            - End with offering assistance if they have questions
            """
        )
        
        st.success("Interview scheduled successfully! Check your email for details.")
        
    except Exception as e:
        logger.error(f"Error scheduling interview: {str(e)}")
        st.error("Unable to schedule interview. Please try again.")


def main() -> None:
    st.set_page_config(page_title="Multi-Agent Recruitment System", layout="wide", initial_sidebar_state="collapsed")
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        /* Main container styling */
        .main {
            padding: 2rem 3rem;
        }
        
        /* Card-like containers */
        .stExpander {
            background-color: #F8FAFC;
            border-radius: 10px;
            border: 1px solid #E2E8F0;
        }
        
        /* Better button styling */
        .stButton>button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
            border: none;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Input field styling */
        .stTextInput>div>div>input {
            border-radius: 8px;
            border: 2px solid #E2E8F0;
            padding: 0.75rem;
            font-size: 1rem;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: #3B82F6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }
        
        /* File uploader styling */
        .stFileUploader {
            background-color: #F8FAFC;
            border-radius: 10px;
            padding: 1.5rem;
            border: 2px dashed #CBD5E1;
        }
        
        /* Success/Info/Warning boxes */
        .stSuccess, .stInfo, .stWarning {
            border-radius: 8px;
            padding: 1rem;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #1E293B;
            font-weight: 600;
        }
        
        /* Status containers */
        .stStatus {
            border-radius: 8px;
        }
        
        /* Select box */
        .stSelectbox>div>div {
            border-radius: 8px;
        }
        
        /* Download button */
        .stDownloadButton>button {
            background-color: #F8FAFC;
            color: #1E293B;
            border: 2px solid #E2E8F0;
        }
        
        .stDownloadButton>button:hover {
            background-color: #E2E8F0;
            border-color: #CBD5E1;
        }
        
        /* Divider */
        hr {
            margin: 2rem 0;
            border: none;
            border-top: 2px solid #E2E8F0;
        }
        
        /* Footer */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #F8FAFC;
            color: #64748B;
            text-align: center;
            padding: 1rem;
            font-size: 0.9rem;
            border-top: 1px solid #E2E8F0;
            z-index: 999;
        }
        
        .footer a {
            color: #3B82F6;
            text-decoration: none;
            font-weight: 500;
        }
        
        .footer a:hover {
            text-decoration: underline;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header with logo and company name together
    logo_col, title_col, spacer_col = st.columns([1, 3, 1])
    
    with logo_col:
        # Logo and company name side by side with vertical alignment
        brand_col1, brand_col2 = st.columns([1, 3])
        with brand_col1:
            try:
                # Add margin to push logo down to align with text
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                st.image("assets/logo_main.png", width=60)
            except:
                pass
        with brand_col2:
            st.markdown("""
                <div style='margin-top: 23px;'>
                    <a href='https://allometrik.com' target='_blank' style='text-decoration: none;'>
                        <h2 style='margin: 0; color: #1E293B; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.5px;'>allometrik</h2>
                    </a>
                </div>
            """, unsafe_allow_html=True)
    
    with title_col:
        st.markdown("""
            <div style='text-align: center; margin-top: 5px;'>
                <h1 style='margin: 0; color: #1E293B; font-size: 2.5rem; font-weight: 600;'>Multi-Agent Recruitment System</h1>
                <p style='color: #64748B; font-size: 1.1rem; margin-top: 0.5rem;'>Intelligent candidate screening and interview scheduling</p>
            </div>
        """, unsafe_allow_html=True)
    
    with spacer_col:
        st.write("")  # Empty spacer for balance
    
    st.markdown("---")

    init_session_state()

    # Top section: Role selection and new application button
    col_role, col_button = st.columns([3, 1])
    with col_role:
        role_display = st.selectbox("Select the role you're applying for:", list(ROLE_DISPLAY_NAMES.keys()))
        role = ROLE_DISPLAY_NAMES[role_display]
    with col_button:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("New Application", use_container_width=True):
            keys_to_clear = ['resume_text', 'analysis_complete', 'is_selected', 'candidate_email', 'current_pdf']
            for key in keys_to_clear:
                if key in st.session_state:
                    st.session_state[key] = None if key == 'current_pdf' else ""
            st.rerun()

    # Role requirements
    with st.expander("View Required Skills", expanded=False):
        st.markdown(ROLE_REQUIREMENTS[role])
    
    st.markdown("---")

    # Main layout: Two columns
    main_col1, main_col2 = st.columns([1, 1], gap="large")

    with main_col1:
        st.markdown("### Application Form")
        st.markdown("")  # Add spacing
        
        # File uploader
        resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"], key="resume_uploader")
        
        if resume_file is not None and resume_file != st.session_state.get('current_pdf'):
            st.session_state.current_pdf = resume_file
            st.session_state.resume_text = ""
            st.session_state.analysis_complete = False
            st.session_state.is_selected = False
            st.rerun()

        # Process the resume text
        if resume_file and not st.session_state.resume_text:
            with st.spinner("Processing your resume..."):
                resume_text = extract_text_from_pdf(resume_file)
                if resume_text:
                    st.session_state.resume_text = resume_text
                    
                    # Try to extract email from resume
                    with st.spinner("Extracting email address..."):
                        extracted_email = extract_email_from_resume(resume_text)
                        if extracted_email:
                            st.session_state.candidate_email = extracted_email
                            st.success(f"Resume processed successfully! Email found: {extracted_email}")
                        else:
                            st.success("Resume processed successfully! Please enter your email address below.")
                else:
                    st.error("Could not process the PDF. Please try again.")

        # Email input
        email = st.text_input(
            "Email Address",
            value=st.session_state.candidate_email,
            key="email_input",
            placeholder="your.email@example.com",
            help="Email was automatically extracted from your resume. You can edit it if needed."
        )
        st.session_state.candidate_email = email
        
        # Auto-analyze when resume and email are available
        if st.session_state.resume_text and email and not st.session_state.analysis_complete:
            st.markdown("---")
            with st.spinner("Analyzing your resume..."):
                resume_analyzer = create_resume_analyzer()
                email_agent = create_email_agent()
                
                if resume_analyzer and email_agent:
                    print("DEBUG: Starting resume analysis")
                    is_selected, feedback = analyze_resume(
                        st.session_state.resume_text,
                        role,
                        resume_analyzer
                    )
                    print(f"DEBUG: Analysis complete - Selected: {is_selected}, Feedback: {feedback}")

                    if is_selected:
                        st.session_state.analysis_complete = True
                        st.session_state.is_selected = True
                        st.rerun()
                    else:
                        st.warning("Unfortunately, your skills don't match our requirements.")
                        with st.expander("View Feedback", expanded=True):
                            st.write(feedback)
                        
                        # Send rejection email
                        with st.spinner("Sending feedback email..."):
                            try:
                                send_rejection_email(
                                    email_agent=email_agent,
                                    to_email=email,
                                    role=role,
                                    role_display=role_display,
                                    feedback=feedback
                                )
                                st.info("We've sent you an email with detailed feedback.")
                            except Exception as e:
                                logger.error(f"Error sending rejection email: {e}")
                                st.error("Could not send feedback email. Please try again.")
        
        # Show next steps if selected
        if st.session_state.get('analysis_complete') and st.session_state.get('is_selected', False):
            st.markdown("---")
            st.markdown("### Candidate Selected:")
            st.success("Congratulations! Skills match the requirements.")
            st.info("Click 'Proceed with Application' to schedule the interview and send confirmation emails.")
            
            if st.button("Proceed with Application", type="primary", use_container_width=True, key="proceed_button"):
                print("DEBUG: Proceed button clicked")
                with st.spinner("Processing your application..."):
                    try:
                        print("DEBUG: Creating email agent")
                        email_agent = create_email_agent()
                        print(f"DEBUG: Email agent created: {email_agent}")
                        
                        print("DEBUG: Creating scheduler agent")
                        scheduler_agent = create_scheduler_agent()
                        print(f"DEBUG: Scheduler agent created: {scheduler_agent}")

                        # Send selection email
                        with st.status("Sending confirmation email...", expanded=True) as status:
                            print(f"DEBUG: Attempting to send email to {st.session_state.candidate_email}")
                            send_selection_email(
                                email_agent,
                                st.session_state.candidate_email,
                                role,
                                role_display
                            )
                            print("DEBUG: Email sent successfully")
                            status.update(label="Confirmation email sent!")

                        # Schedule interview
                        with st.status("Scheduling interview...", expanded=True) as status:
                            print("DEBUG: Attempting to schedule interview")
                            schedule_interview(
                                scheduler_agent,
                                st.session_state.candidate_email,
                                email_agent,
                                role,
                                role_display
                            )
                            print("DEBUG: Interview scheduled successfully")
                            status.update(label="Interview scheduled!")

                        print("DEBUG: All processes completed successfully")
                        
                        # Success message with better styling
                        st.markdown("""
                            <div style='background-color: #F0FDF4; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #22C55E; margin: 1rem 0;'>
                                <h3 style='color: #166534; margin: 0 0 0.5rem 0;'>Application Successfully Processed!</h3>
                                <p style='color: #15803D; margin: 0;'>All communications have been sent to the candidate.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("**Emails Sent:**")
                        st.markdown("- Selection confirmation")
                        st.markdown("- Interview details with Microsoft Teams link")
                        
                        # Generate interview questions for the recruiter
                        st.markdown("---")
                        st.markdown("### Suggested Interview Questions")
                        
                        with st.spinner("Generating personalized interview questions..."):
                            questions = generate_interview_questions(
                                st.session_state.resume_text,
                                role,
                                role_display
                            )
                            
                            if questions:
                                for i, question in enumerate(questions, 1):
                                    st.markdown(f"""
                                        <div style='background-color: #F8FAFC; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 3px solid #3B82F6;'>
                                            <strong style='color: #3B82F6;'>Question {i}:</strong><br/>
                                            <span style='color: #1E293B;'>{question}</span>
                                        </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("Could not generate questions at this time.")

                    except Exception as e:
                        print(f"DEBUG: Error occurred: {str(e)}")
                        print(f"DEBUG: Error type: {type(e)}")
                        import traceback
                        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
                        st.error(f"An error occurred: {str(e)}")
                        st.error("Please try again or contact support.")

    with main_col2:
        if resume_file:
            st.markdown("### Resume Preview")
            st.markdown("")  # Add spacing
            
            import tempfile, os
            resume_file.seek(0)  # Reset file pointer before reading
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(resume_file.read())
                tmp_file_path = tmp_file.name
            resume_file.seek(0)  # Reset again for download button
            
            # Add a container for the PDF viewer
            with st.container():
                try: 
                    pdf_viewer(tmp_file_path)
                finally: 
                    os.unlink(tmp_file_path)
            
            st.markdown("")  # Add spacing
            st.download_button(
                label="⬇ Download Resume", 
                data=resume_file, 
                file_name=resume_file.name, 
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.markdown("### Resume Preview")
            st.markdown("")
            st.info("Upload a resume to see the preview here")
    
    # Footer with credits
    st.markdown("<br><br><br>", unsafe_allow_html=True)  # Add space for fixed footer
    st.markdown("""
        <div class='footer'>
            <p style='margin: 0;'>
                Built with ❤️ by 
                <a href='https://allometrik.com' target='_blank'><strong>Allometrik</strong></a>
                — Specialized in building cutting-edge agentic AI solutions
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()