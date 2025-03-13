import json
import os
from flask import Flask, request, jsonify
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API credentials from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_ENDPOINT")

# Initialize Flask app
app = Flask(__name__)

# Load the dataset of contract prompts and outputs
try:
    with open("contract_prompts.json", "r") as f:
        contract_data = json.load(f)
except Exception as e:
    print("Error loading contract_prompts.json:", e)
    contract_data = []

# Initialize Azure OpenAI model (GPT-4)
llm = AzureChatOpenAI(
    deployment_name="gpt-4-1106",
    model_name="gpt-4",
    temperature=0,  # Set to 0 for deterministic responses
    openai_api_key=openai_api_key,
    azure_endpoint=azure_endpoint,
    openai_api_type="azure",
)

def search_contract_in_dataset(prompt):
    """
    Searches the dataset for a pre-generated contract matching the prompt.
    """
    for entry in contract_data:
        if prompt.lower() in entry["prompt"].lower():
            return entry["output"]
    return None

def llm_generate_contract(prompt):
    """
    Uses GPT-4 via Azure OpenAI API to generate Solidity contract code.
    """
    print(f"Generating Solidity contract for prompt: {prompt}")

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "// Error generating Solidity contract"

@app.route('/generate_contract', methods=['POST'])
def generate_contract():
    """
    API Endpoint: Accepts a text description (prompt) and returns a Solidity contract.
    """
    data = request.get_json(force=True)
    if "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' in request body"}), 400

    prompt = data["prompt"]

    # Check dataset for a matching prompt
    contract_code = search_contract_in_dataset(prompt)
    
    if contract_code is None:
        # If no match found, generate contract using GPT-4
        contract_code = llm_generate_contract(prompt)

    return jsonify({"solidity_contract": contract_code})

@app.route('/status', methods=['GET'])
def status():
    """
    API Endpoint: Returns the API status and dataset details.
    """
    return jsonify({
        "status": "Solidity Contract Generator API is running",
        "dataset_entries": len(contract_data)
    })

if __name__ == '__main__':
    app.run(debug=True)
