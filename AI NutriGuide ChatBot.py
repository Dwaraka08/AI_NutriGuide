import streamlit as st
import os
import google.generativeai as genai
import litellm
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel
import speech_recognition as sr
import pyttsx3
from PIL import Image

# Enable debug
litellm._turn_on_debug()

# Set API key and configure
os.environ["GOOGLE_API_KEY"] = "AIzaSyApCfL9HdN1qb9CWr1WVyz3xk-4FA6Ygs4"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize models
model_text = LiteLLMModel(model_id="gemini/gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
#model_image = LiteLLMModel(model_id="gemini/gemini-pro-vision", api_key=os.getenv("GOOGLE_API_KEY"))

# Setup text agent
ml_code_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    additional_authorized_imports=['pandas', 'numpy', 'sklearn', 'json'],
    model=model_text
)

# Initialize recognizer and TTS
r = sr.Recognizer()

def speach(text):
    engine = pyttsx3.init()  # moved inside function
    engine.say(text)
    engine.runAndWait()

def recognition_speech():
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Listening....")
        audio = r.listen(source)
        try:
            command = r.recognize_google(audio)
            print("Your command:", command)
            return command
        except sr.UnknownValueError:
            speach("Sorry, I couldn't understand your message.")
        except sr.RequestError as e:
            speach("Sorry, there was a request error.")
            print("Request error:", e)

def build_nutriguide_prompt(user_input, health_info=None, fitness_info=None, preferences=None, feedback=None):
    base_prompt = f"""
You are NutriGuide — a highly intelligent, caring, and adaptive AI nutritionist.

Your core mission is to understand the user's intention — whether they are asking a simple question about food, or requesting a full personalized meal plan — and respond accordingly.

Always respond in clear, concise, and well-structured plain text suitable for end users to read comfortably or save as a text file.

- Use simple paragraphs, bullet points, or numbered lists where appropriate.
- Avoid JSON, code blocks, tables, or any formatting that is hard to read in plain text.
- Avoid overly technical language or verbose introductions.
- Do not include apologies or unnecessary filler text.

---

If the user's question is NOT related to health, diet, or nutrition topics, reply politely and clearly with:

"Sorry, I can only assist with health, diet, or nutrition-related questions."

---

## If the user asks a **basic question** about health, diet, or nutrition:
- Answer clearly and concisely.
- Add **contextual explanation** to support your response.
- If relevant, offer **general food tips or alternatives** that align with their health condition or goal.

## If the user seems to ask for a **personalized diet/plan** about health, diet, or nutrition:
- Generate a full **7-day structured meal plan** in a clear, easy-to-read text format like:

Monday  
- Breakfast: ...  
- Lunch: ...  
- Dinner: ...  
- Snacks: ...

- Do NOT output JSON, code blocks, or complex formatting.
- Include portion sizes and nutrition-conscious choices.
- Adapt the plan using:
  - Health conditions (e.g., diabetes, PCOS, cholesterol)
  - Fitness goals (e.g., muscle gain, weight loss)
  - Food preferences/restrictions (e.g., vegetarian, dairy-free)
  - Feedback (e.g., "I felt bloated with lentils")

- After the plan, include a short **explanation of why this plan fits the user.**

---

### User Request:
{user_input}

### Health Conditions / Goals:
{health_info or "Not provided"}

### Fitness Info:
{fitness_info or "Not provided"}

### Food Preferences or Restrictions:
{preferences or "Not specified"}

### User Feedback:
{feedback or "None"}

---

Think like a real expert dietitian who adjusts advice based on user intention. Return:
- A direct answer for basic queries, **OR**
- A 7-day meal plan + explanation if the user needs a full plan,
- OR the polite refusal message above if the question is unrelated to health, diet, or nutrition.

Do not include apologies or unnecessary introductions.
"""
    return base_prompt


def process_text(user_input, health_info=None, fitness_info=None, preferences=None, feedback=None):
    prompt = build_nutriguide_prompt(user_input, health_info, fitness_info, preferences, feedback)
    try:
        return ml_code_agent.run(prompt)
    except Exception as e:
        return f"Error: {str(e)}"

def process_image(image_file, health_info=None, fitness_info=None, preferences=None, feedback=None):
    try:
        image = Image.open(image_file)
        vision_model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
You are NutriGuide — a highly intelligent, caring, and adaptive AI nutrition assistant specialized in interpreting food images, grocery labels, and nutrition reports.

Analyze the uploaded image carefully:

Do NOT output JSON, code blocks, or complex formatting.
- If it's a **meal photo**, identify all visible food items, estimate calorie and nutrient content, and assess the overall healthiness of the meal considering the user's health profile and mention for which health condition people this food is suitable .
- If it's a **nutrition label or grocery item**, summarize key nutritional information clearly and evaluate if it suits the user's health goals and dietary restrictions and mention for which health condition this food is suitable .

- Do NOT output JSON, code blocks, or complex formatting.
- Include portion sizes and nutrition-conscious choices.
Use the following user data to tailor your analysis and recommendations:
- Health Conditions / Goals: {health_info or "Not provided"}
- Fitness Info: {fitness_info or "Not provided"}
- Food Preferences or Restrictions: {preferences or "Not specified"}
- User Feedback: {feedback or "None"}

Provide:
- Clear, concise, and readable nutrition advice and analysis.
- Personalized suggestions or warnings relevant to their health and goals.
- If applicable, suggest how this food or meal fits into a healthy diet plan.

If the image content or query is unrelated to health, diet, or nutrition, respond politely with:
"Sorry, I can only assist with health, diet, or nutrition-related questions."

Do not include apologies or unnecessary introductions. Present the response in a user-friendly plain text format suitable for reading or saving.

"""
        response = vision_model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Image processing error: {str(e)}"


def create_downloadable_file(content):
    try:
        with open("nutrition_advice.txt", "w") as file:
            file.write(content)
        return "nutrition_advice.txt"
    except Exception as e:
        return f"File creation error: {str(e)}"

# -----------------------------
# Streamlit UI (Chat-style)
# -----------------------------
st.set_page_config(page_title="NutriGuide Chat", page_icon="🥗")
st.title("🥗 NutriGuide - Your AI Nutrition Assistant")

# Sidebar
with st.sidebar:
    st.header("🧠 About NutriGuide")
    st.write("NutriGuide gives personalized nutrition advice")
    uploaded_file = st.file_uploader("📷 Upload food image or label", type=["jpg", "jpeg", "png"])
    st.info("Ask the question related to your diet and nutrition advice")
    if st.button("🗑️ Clear Chat"):
        st.session_state.pop("messages", None)

# Session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hi! I'm NutriGuide 🥗. Ask me anything about healthy eating or upload a food image."
    }]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_prompt = st.chat_input("Ask a nutrition question...")
col1, col2 = st.columns([0.8, 0.2])

with col2:
    if st.button("🎤Use Microphone"):
        st.toast("Listening.....")
        speech_input = recognition_speech()
        if speech_input:
            user_prompt = speech_input
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            #with st.chat_message("user"):
                #st.write(user_prompt)

# Main response
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("NutriGuide is thinking..."):
            result = process_image(uploaded_file) if uploaded_file else process_text(user_prompt)
            st.write(result)
        st.session_state.messages.append({"role": "assistant", "content": result})

        file_path = create_downloadable_file(result)
        st.download_button(
            label="⬇️ Download Advice",
            data=open(file_path, "rb").read(),
            file_name="nutrition_advice.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.info("💡 Tip: Upload a food photo or ask questions like 'What to eat post-workout?' or 'Low-carb lunch ideas'.")
