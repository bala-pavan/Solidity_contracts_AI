from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import getpass
import json

# Load environment variables
load_dotenv()

# Set API key if not found in environment
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# Initialize LangChain LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Define the prompt
prompt = """
Generate a JSON file containing 10 Solidity smart contract samples. 
Each entry should have a 'prompt' describing the contract type and an 'output' containing the Solidity code. 
The contracts should follow OpenZeppelin standards where applicable.
The JSON structure should look like this:

[
  {
    "prompt": "Generate an ERC-20 token contract with burn functionality",
    "output": "pragma solidity ^0.8.0;\ncontract MyToken { ... }"
  },
  {
    "prompt": "Generate an ERC-721 NFT contract with metadata storage",
    "output": "pragma solidity ^0.8.0;\ncontract MyNFT { ... }"
  }
]

Ensure the Solidity code is well-formatted, and each contract implements its intended functionality properly.
"""

# Generate contract samples
response = llm.invoke(prompt)  # Use invoke() instead of generate_content()
solidity_samples = response.content  # Extract response

# Try parsing the JSON output
try:
    samples = json.loads(solidity_samples)
except json.JSONDecodeError:
    print("Error: Failed to parse JSON output.")
    samples = []

file_path = "contract_prompts.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=4)

print(f"✅ Solidity contract samples saved to: {file_path}")