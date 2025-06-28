# app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests
import json
import uvicorn # Used for local development and explicit server start

# Load environment variables from .env file (for local development)
load_dotenv()

# Initialize FastAPI application
app = FastAPI(
    title="Marshee Pet Tech Dog Product Recommender ",
    description="AI powered product recommender for dog owners.",
)

# Mount the 'templates' directory to serve static files.
# This makes 'templates/index.html' accessible at the root URL '/'.
app.mount("/static", StaticFiles(directory="templates"), name="static")

# Retrieve Groq API key from environment variable
# On Render, this will be set in your service's environment variables.
# Locally, it will come from your .env file.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    # Raise an error if API key is not found, to prevent silent failures
    raise ValueError(
        "GROQ_API_KEY environment variable not set. "
        "Please ensure it's in your .env file locally or set on Render."
    )

# Pydantic model to define the expected structure of incoming request data
class RecommendationRequest(BaseModel):
    dog_breed: str
    diet_preference: str
    product_type: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Serves the main HTML page for the dog product recommender demo.
    """
    # Read the index.html file and return its content
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/get_recommendation")
async def get_recommendation(request_data: RecommendationRequest):
    """
    Handles the recommendation request.
    It takes dog details, constructs a prompt, sends it to the Groq LLM,
    and returns a personalized product recommendation and insight.
    """
    # Initialize llm_response_content for logging in case of parsing errors
    llm_response_content = ""
    try:
        dog_breed = request_data.dog_breed.strip()
        diet_preference = request_data.diet_preference
        product_type = request_data.product_type.strip()

        # Construct the prompt for the Groq LLM
        prompt = f"""
        As an AI assistant for a dog product company, provide a brief, personalized product recommendation and a quick, relevant insight for a dog parent.

        Dog Breed: {dog_breed}
        Dietary Preference: {diet_preference}
        Desired Product Type: {product_type}

        Please provide:
        1. A specific, plausible product recommendation (e.g., 'XYZ Brand Organic Chicken Dog Food').
        2. A concise insight related to cost-benefit or community trend (e.g., '80% of Golden Retriever owners prefer large bags for cost savings.' or 'This toy is a hit with active breeds!').

        Format your response strictly as a JSON object with two keys: "recommendation" and "insight".
        Example:
        {{
          "recommendation": "Durable Chew Toy for Large Breeds",
          "insight": "Chew toys help reduce anxiety in high-energy dogs."
        }}
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }

        payload = {
            "model": "llama3-8b-8192", # Using an efficient open-source model available on Groq
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "response_format": { "type": "json_object" }, # Crucial for getting structured JSON from LLM
            "temperature": 0.7 # Controls creativity; 0.7 is a good balance
        }

        # Make the API call to Groq
        groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
        groq_response = requests.post(groq_api_url, headers=headers, json=payload)
        groq_response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

        groq_data = groq_response.json()
        
        # Extract the content from the LLM's response
        llm_response_content = groq_data['choices'][0]['message']['content']
        parsed_llm_content = json.loads(llm_response_content) # Parse the JSON string from LLM

        recommendation = parsed_llm_content.get('recommendation')
        insight = parsed_llm_content.get('insight')

        if recommendation and insight:
            return {"recommendation": recommendation, "insight": insight}
        else:
            # If LLM doesn't follow format, log and return an error
            print(f"WARNING: LLM response missing expected keys. Raw content: {llm_response_content}")
            raise HTTPException(
                status_code=500,
                detail="Recommendation service encountered an issue. Please try again."
            )

    except requests.exceptions.RequestException as e:
        # Handle errors during the API call to Groq
        print(f"ERROR: Groq API call failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to recommendation service (Groq API error). Error: {e}"
        )
    except json.JSONDecodeError as e:
        # Handle cases where LLM response is not valid JSON
        print(f"ERROR: Failed to parse LLM JSON response: {e}. Raw content: {llm_response_content}")
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation service received invalid data. Error: {e}"
        )
    except Exception as e:
        # Catch any other unexpected errors
        print(f"ERROR: An unexpected server error occurred: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected server error occurred: {e}"
        )

if __name__ == '__main__':
    # Get the port from the environment variable (Render sets this) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    # Run the Uvicorn server, binding to 0.0.0.0 for external access
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True) # reload=True for local dev