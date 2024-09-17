import streamlit as st
import boto3
import json
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def initialize_bedrock_client():
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION")
    
    if not all([aws_access_key_id, aws_secret_access_key, aws_region]):
        st.error("AWS credentials or region not found. Please set them in your environment or .env file.")
        st.stop()

    try:
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        client = session.client('bedrock-runtime')
        return client
    except Exception as e:
        st.error(f"Failed to initialize AWS Bedrock client: {str(e)}")
        st.stop()

def bedrock_llama_response(client, informal_statement):
    model_id = os.getenv("BEDROCK_MODEL_ID")
    if not model_id:
        st.error("BEDROCK_MODEL_ID not found. Please set it in your environment or .env file.")
        st.stop()

    prompt = f"""As an expert in mathematical theorem formalization, your task is to convert the given informal mathematical statement into a formal theorem statement in Lean. Follow these guidelines:

1. Use proper Lean syntax and notation.
2. Include appropriate type declarations for variables.
3. Use Unicode characters for mathematical symbols where applicable.
4. Structure the theorem statement logically, using implications (→) or biconditionals (↔) as needed.
5. If the statement involves multiple conditions or parts, use appropriate logical connectives (∧, ∨, ¬).
6. For statements about sets or types, use appropriate quantifiers (∀, ∃) and set notation.
7. If the statement involves specific mathematical concepts (e.g., primality, divisibility), use the corresponding Lean functions or definitions.

Informal statement: {informal_statement}

Formal Lean theorem:"""
    try:
        body = json.dumps({
            "prompt": prompt,
            "max_gen_len": 300,
            "temperature": 0.2,
            "top_p": 0.95,
        })

        response = client.invoke_model(
            body=body,
            modelId=model_id,
            accept='application/json',
            contentType='application/json'
        )

        response_body = json.loads(response.get('body').read())
        formal_statement = response_body.get('generation')
        return formal_statement.strip() if formal_statement else "Conversion failed."
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        st.error(f"AWS Bedrock API error: {error_code} - {error_message}")
    except Exception as e:
        st.error(f"An error occurred while calling the AWS Bedrock API: {str(e)}")
    return None

def main():
    st.set_page_config(page_title="Math Theorem Converter (AWS Bedrock)", page_icon="🧮")

    st.title("Informal to Formal Math Theorem Converter")

    st.write("This app converts informal mathematical statements to formal theorem statements in Lean using AWS Bedrock's Llama model.")

    client = initialize_bedrock_client()

    informal_statement = st.text_area("Enter informal mathematical statement:", height=100,
                                      placeholder="e.g., The cardinality of the antidiagonal of n is n+1.")

    if st.button("Convert"):
        if informal_statement:
            with st.spinner("Converting..."):
                formal_statement = bedrock_llama_response(client, informal_statement)
                if formal_statement:
                    st.subheader("Formal Theorem Statement:")
                    st.code(formal_statement, language="lean")
        else:
            st.warning("Please enter a mathematical statement to convert.")

    st.markdown("---")
    st.subheader("Sample Inputs:")
    samples = [
        "The golden ratio is irrational.",
        "There are no perfect squares strictly between m² and (m+1)²",
        "The only numbers with empty prime factorization are 0 and 1",
        "Odd Bernoulli numbers (greater than 1) are zero.",
        "A natural number is odd iff it has residue 1 or 3 mod 4"
    ]
    for sample in samples:
        if st.button(f"Try: {sample}"):
            st.text_area("Informal statement:", value=sample, key=sample)

if __name__ == "__main__":
    main()