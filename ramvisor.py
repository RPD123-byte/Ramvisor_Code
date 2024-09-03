from flask import Flask, request, jsonify, render_template, session
from flask_session import Session  # You may need to install this with pip
import openai
import ast
import traceback  # Import traceback for detailed error information
import json
import weaviate
import re
from transformers import GPT2Tokenizer
import requests

# AWS_KEY = "AKIARCDY2XIZ6O72YRAO"
# AWS_SECRET = "Yjkkg69chm5Zj5tGiNvex/RANwHVB1qnTd6+3y8yaws"

openai.api_key = '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'

# Initialize Flask application
app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
# app.config['SESSION_FILE_DIR'] = 'your_session_storage_path'  # Define the path where session files will be saved
app.config["SESSION_TYPE"] = "filesystem"
app.config['UPLOAD_FOLDER'] = 'user_uploads'  # Make sure this directory exists or create it
Session(app)

# app.secret_key = 'fopg928e0jvc1eibpvqoicnqinc'


@app.route('/')
def home():
    return render_template('chat.html')

def enhance_question_with_context(previous_questions, new_question):
    # If there are no previous questions or the new question is already specific, return the original question
    if not previous_questions or not new_question.strip():
        return new_question

    # Create a prompt for the model to add only essential context
    prompt = """
    Given a list of previous questions and a new question, add only the most essential context to the new question without altering its structure or specificity. The aim is to retain the question's original intent and only include additional keywords that relate directly to the subject matter, without introducing new elements about timing or requirements unless they are crucial to the context established by the previous questions. The additions you make should amount to a max total of 3 words.
    Previous Questions:
    """
    for question in previous_questions:
        prompt += f"- {question}\n"

    prompt += f"""
    New Question: {new_question}
    Provide the Enhanced Question with only the most crucial context added if no change is needed then keep the question the same as the original:
    """

    # Send the prompt to the OpenAI API
    response = openai.Completion.create(
        engine="text-davinci-003",  # Use the latest available engine
        prompt=prompt,
        max_tokens=50  # Adjust tokens as needed
    )

    # Extract the enhanced question from the response
    enhanced_question = response.choices[0].text.strip()

    return enhanced_question




class HierarchicalChunkProcessor:
    def __init__(self):
        self.data = {}

    def load_from_json_files(self, json_paths):
        for json_path in json_paths:
            with open(json_path, 'r') as file:
                data_chunk = json.load(file)
                self.data.update(data_chunk)

    def format_chunks(self):
        results = []
        for heading, sub_content_list in self.data.items():
            # Process each item in the sub_content_list
            for item in sub_content_list:
                # Check if the item is a dictionary with subheadings
                if isinstance(item, dict):
                    for sub_key, sub_values in item.items():
                        # If sub_values is a list, it might contain more dictionaries
                        if isinstance(sub_values, list):
                            for value in sub_values:
                                # If the value is a dictionary, it's another subheading
                                if isinstance(value, dict):
                                    for third_level_key, third_level_values in value.items():
                                        if isinstance(third_level_values, list):
                                            # If the third-level values are a list, iterate and format each one
                                            for val in third_level_values:
                                                results.append(f"{heading} - {sub_key} - {third_level_key}: {val}")
                                        else:
                                            # If it's a single value, format it directly
                                            results.append(f"{heading} - {sub_key} - {third_level_key}: {third_level_values}")
                                else:
                                    # If it's just a string, append it with its heading and subheading
                                    results.append(f"{heading} - {sub_key}: {value}")
                        else:
                            # If sub_values is just a string, append it with its heading
                            results.append(f"{heading} - {sub_key}: {sub_values}")
                else:
                    # If the item is just a string, append it with its main heading
                    results.append(f"{heading}: {item}")
        return results
    
def extract_heading(text_chunk):
    # The heading is defined as the part before the first colon
    return text_chunk.split(':')[0]

def contains_keyword(heading, keywords):
    # Check if any keyword is in the heading
    return any(keyword.lower() in heading.lower() for keyword in keywords)


 
def vector_search(user_input, given_property):
    # Configure the client to connect to your Weaviate instance
    auth_config = weaviate.AuthApiKey(api_key="lPMSnt78RQzC2LNRFEwJWrLUopCzeJ5h9tPi")

    client = weaviate.Client(
        url="https://ramvisor-sbwhtuuh.weaviate.network",
        auth_client_secret=auth_config,
        additional_headers={
            "X-OpenAI-Api-Key": "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",
        }
    )

    # Perform a vector search query to find text chunks related to the input, focusing on headings
    nearText = {
        "concepts": [user_input],
        "properties": [given_property]  # Focus on the combined headings for the search
    }

    response = (
        client.query
        .get("TextChunk", ["main_heading", "subheading", "final_subheading", "content"])
        # .with_near_text(nearText)
        .with_hybrid(
            query = user_input,
            alpha = 0.8
        )
        .with_limit(20)  # Adjust the limit as needed to return more results
        .with_additional("certainty")  # Request the similarity scores (certainty)
        .do()
    )

    # Process the response to format it as two separate lists: one for text chunks, one for similarity scores
    formatted_results = []

    for item in response['data']['Get']['TextChunk']:
        # Reconstruct the text chunk with headings and content
        text_chunk_headings = ''
        if item.get('main_heading'):
            text_chunk_headings += item['main_heading']
        if item.get('subheading'):
            text_chunk_headings += ' - ' + item['subheading']
        if item.get('final_subheading'):
            text_chunk_headings += ' - ' + item['final_subheading']
        if item.get('content'):
            text_chunk_headings += ':' + item['content']
        formatted_results.append(text_chunk_headings)
    
    keywords = ["FY-SEMINAR", "FY_LAUNCH", "GLBL-LANG", "FC-AESTH", "FC-CREATE", "FC-PAST", "FC-VALUES", "FC-GLOBAL-FC-NATSCI", "FC-POWER", "FC-QUANT", "FC-KNOWING", "FC-LAB"]

    # Rerank the results based on specific keywords in the headings
    reranked_results = sorted(
        formatted_results,
        key=lambda x: any(keyword in x.split(':', 1)[0] for keyword in keywords),
        reverse=True
    )

    # Keep only the top 5 results after reranking
    top_results = reranked_results[:5]

    return top_results

def is_follow_up(new_question, previous_questions):
    if not previous_questions:  # If there are no previous questions, return False
        return False

    # Create a prompt for the model with strict criteria for follow-up
    prompt = """
    I will provide a series of questions. You must determine if the last question is contextually related or a direct follow-up to any of the previous ones. Consider a question a follow-up only if it directly references specific information or a topic mentioned in the previous questions without any ambiguity.

    Previous Questions:
    """
    for i, question in enumerate(previous_questions, 1):
        prompt += f"Q{i}: {question}\n"

    prompt += f"""
    New Question: {new_question}
    Based on the strict criteria provided, is the New Question a direct and obvious follow-up? Answer 'yes' only if the connection is unambiguous and clear.
    """

    # Send the prompt to the OpenAI API
    response = openai.Completion.create(
        engine="text-davinci-003",  # or the latest available engine
        prompt=prompt,
        max_tokens= 5 # Adjust tokens as needed
    )

    # Interpret the response
    answer = response.choices[0].text.strip().lower()
    return "yes" in answer  # Return true if the model's response is clearly "yes"


def is_schedule_planning_question(question):
    # Define a prompt that describes the task to the model
    prompt = f"""
    Determine if the following question is specifically about planning a course schedule, such as choosing courses for a semester to meet academic requirements, or if it's a general question about course registration that does not involve planning a schedule. Respond with 'schedule planning' for the former or 'general registration' for the latter.

    Question: "{question}"

    Classification:"""

    # Send the prompt to the OpenAI API
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=10  # Limit the tokens to get a precise classification
    )

    # Interpret the response
    classification = response.choices[0].text.strip().lower()
    
    # Check if the classification is related to schedule planning
    return "schedule planning" in classification



def get_response_streamed(conversation_history, system_message, user_prompt, temperature, frequency_penalty, presence_penalty, top_p):
    # Append new system and user messages to the message history
    conversation_history.append({"role": "system", "content": system_message})
    conversation_history.append({"role": "user", "content": user_prompt})
    
    while calculate_token_count(conversation_history) > 16384:
        conversation_history.pop(0) 
    
    # Initialize the stream
    response_stream = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=conversation_history,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        top_p=top_p,
        stream=True
    )
    
    # Collect the response parts as they come in
    full_response_content = ""
    for response in response_stream:
        if 'choices' in response and len(response['choices']) > 0:
            choice = response['choices'][0]
            if 'delta' in choice and 'content' in choice['delta']:
                content = choice['delta']['content']
                full_response_content += content
                print(content, end='', flush=True)  # Print the content as it's generated
            else:
                # Print only if there is an error key
                if 'error' in choice:
                    print(f"Error: {choice['error']}")
        else:
            # If response is not the expected structure, you can print it out or handle it as you like
            print("Received an unexpected response:", response)
    
    # Return the full response content and the updated message history
    return full_response_content, conversation_history

def calculate_token_count(conversations):
    return sum([len(tokenizer.encode(entry['content'])) for entry in conversations])

# Function to trim the conversation history if the token limit is exceeded
def trim_conversation_history(conversation_history, max_tokens):
    while calculate_token_count(conversation_history) > max_tokens:
        # Remove the oldest messages first (both user and assistant to keep the flow)
        if len(conversation_history) > 1:
            conversation_history.pop(0)  # Remove the oldest user message
            conversation_history.pop(0)  # Remove the oldest assistant message
        else:
            # If there's only one message left and it's still too long, clear the history
            conversation_history.clear()
            break


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')



def get_botpress_conversation_history(bot_id, personal_access_token):
    # Botpress API endpoint to get conversations
    api_endpoint = f"https://api.botpress.cloud/v1/chat/conversations"
    
    # Headers for the API request
    headers = {
        "Authorization": f"Bearer {personal_access_token}",
        "x-bot-id": bot_id
    }
    
    # Send a GET request to the Botpress API
    response = requests.get(api_endpoint, headers=headers)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Return the JSON response containing the conversations
        return response.json()
    else:
        # Handle errors or unsuccessful response
        print("Failed to retrieve conversations: ", response.status_code)
        return None

# Endpoint to get the AI-generated response for the user's question
@app.route('/ask', methods=['POST'])
def ask():
    try:
        # # Extract the user input from the request
        data = request.json
        print(data)
        user_input = data.get('question', '')
        user_questions = data.get('all_messages', [])
        print(user_questions)

        if 'conversation_history' not in session:
            session['conversation_history'] = []

        # user_questions = session['user_questions']
        conversation_history = session['conversation_history']

        conversation_history.append({"role": "user", "content": user_input})

        print(f"User input: {user_input}")

        print(f"User Questions: {user_questions}")
        if user_questions == "":
            enhanced_question = user_input
        else:
            enhanced_question = enhance_question_with_context(user_questions, user_input)
        print(f"Enhanced Question: {enhanced_question}")
        # session['user_questions'] = user_questions  # Save the updated questions back to the session
        file_list = [
            'all_jsons/chunk_7_10.json', 'all_jsons/chunk_10_13.json', 'all_jsons/chunk_15_17.json', 'all_jsons/chunk_17_19.json', 
            'all_jsons/chunk_20_29.json', 'all_jsons/modified_chunk_30_1032_new.json', 'all_jsons/modified_chunk_1032_1240_new.json', 'all_jsons/chunk_1240_1244.json',
            'all_jsons/chunk_1245_1251.json', 'all_jsons/chunk_1251_1252.json', 'all_jsons/chunk_1252_1254.json', 
            'all_jsons/chunk_1254_1255.json', 'all_jsons/chunk_1255_1256.json', 'all_jsons/chunk_1256_1264.json', 
            'all_jsons/chunk_1264_1270.json', 'all_jsons/chunk_1270_1276.json', 'all_jsons/chunk_1276_1278.json', 
            'all_jsons/chunk_1278_1280.json', 'all_jsons/chunk_1280_1286.json', 'all_jsons/chunk_1286_1287.json', 
            'all_jsons/chunk_1287_1290.json', 'all_jsons/chunk_1290_1291.json'
        ]        
        processor = HierarchicalChunkProcessor()
        processor.load_from_json_files(file_list)
        formatted_chunks = processor.format_chunks()
        formatted_results = vector_search(enhanced_question, "combined_headings")

        for result in formatted_results:
            # Split the result to extract the headings
            headings = result.split(':')[0]  # This gets the text before the colon, which are the headings
            print(headings)
        # for r in formatted_results:
        #     print(r)
        # matched_results = find_chunks_with_headings(formatted_results, formatted_chunks)
        # matched_results = '\n'.join(matched_results)
        matched_results = '\n\n'.join(formatted_results)
        # print(matched_results)
        question_type = is_schedule_planning_question(user_input)
        if question_type == True:
            print("... using course plan LLM")
            prompt = f"""
            Context:
            __________
            {matched_results}
            __________
            Q: {user_input}. 

            A:
            """
            
            system_instruct = """
            When planning a course schedule for any major, it is crucial to integrate the following considerations:

            1. Focus Capacities: Every major requires completion of a course in each of nine focus capacities (Aesthetic and Interpretive Analysis, Creative Expression, Engagement with the Human Past, etc.), each worth 3 credit hours. One of these courses must include or be associated with a one-credit lab, specifically the Empirical Investigation Lab.

            2. Credit Hour Management: Students must complete 120 credit hours in total to graduate. Each academic year consists of two semesters, with students typically enrolling in 12 to 18 credit hours per semester. Special permission is required to exceed this limit.

            3. Detailed Course Analysis: Examine the specific credit hours and requirements for each course, ensuring that every recommended course aligns with the major's requirements and the focus capacities.

            4. Semester Balance: Provide a balanced schedule for each semester, taking into account the workload and prerequisites, while ensuring that the focus capacities are evenly distributed throughout the academic years.

            5. Graduation Pathway: Create a comprehensive plan that outlines a pathway to graduation, ensuring all focus capacities and major-specific requirements are met within the standard timeframe of the degree.

            6. Flexibility for Electives: Include room for elective courses that students may choose based on their interests, ensuring these do not conflict with the completion of required courses.

            7. Lab Requirement: Identify which focus capacity course will include the Empirical Investigation Lab, fulfilling the lab requirement.

            8. Adjustments for Academic Progress: Adjust the schedule based on the student's year of study, ensuring that the course recommendations are appropriate for their current academic standing.

            9. Special Permissions: Acknowledge when special permission might be needed for a student to enroll in more than the standard credit hour limit per semester.

            In your response, provide a detailed semester-by-semester breakdown of courses, including focus capacity courses, major-specific courses, electives, and the lab requirement. Ensure that the plan adheres to the credit hour constraints per semester and totals to the required 120 credit hours for graduation.
            """

            # Example usage with the new parameters and message history
            response_content_2, conversation_history = get_response_streamed(
                conversation_history,
                system_instruct,
                prompt,
                temperature=0.2,
                frequency_penalty=0,
                presence_penalty=0,
                top_p=1
            )

            print(response_content_2)
        else:
            print("... using general question LLM")
            prompt = f"""
           Context:
            __________
            {matched_results}
            __________
            Q: {user_input}

            A: 

            """

            system_instruct = """

            1. Emphasize Headings: Instruct the AI to pay particular attention to headings that match the keywords, as these are likely to contain the most relevant information.

            2. Validate Relevance: After identifying potential answers, the AI should check if the content under the relevant headings actually relates to the query. If not, it should continue the search or notify the user.

            3. Admit Uncertainty: If the AI cannot find a clear match or if the content under the relevant headings does not provide a sufficient answer, it should state its uncertainty and suggest alternative ways for the user to find the necessary information, such as contacting an academic advisor or checking the official course catalog.
            
            4. Identify Keywords: When a question is asked, identify the main keywords, especially course numbers, course names, or specific requirement criteria.

            5. Contextual Search: Use these keywords to search through the provided context data. Look for any mention of these keywords within the course descriptions, prerequisites, or degree requirements.

            6. Reference the Context: When you find a match, reference the exact text from the context that includes the keyword. This text should form the basis of your answer.

            7. Justify with Direct Quotes: Provide direct quotes from the context that pertain to the keywords as justification for the answer you give. If a course number or name is mentioned in relation to specific degree requirements, include this in your response.

            8. Clear and Specific Answers: Your response should clearly state whether the course in question meets the specified requirement, based on the context data where the keyword appears. Justify each of your answers individually and state why you gave them.

            9. Clarify Uncertainties: If the context does not contain clear information or if the keyword search yields no results, inform the user that the question cannot be answered based on the current data and suggest seeking additional information from official sources.

            By focusing on these steps, the system should provide responses that are directly linked to the text's mentions of the keywords found in the user's question, ensuring that the answers are data-driven and verifiable.

            """


            # Example usage with the new parameters and message history
            response_content_2, conversation_history = get_response_streamed(
                conversation_history,
                system_instruct,
                prompt,
                temperature=0.1,
                frequency_penalty=0,
                presence_penalty=0,
                top_p=1
            )
        
        # print(matched_results)
        # user_questions.append(user_input)

        # print(matched_results)

        session['user_questions'] = user_questions
        session['conversation_history'] = conversation_history

        conversation_history.append({"role": "assistant", "content": response_content_2})

        session['conversation_history'] = conversation_history

        return jsonify({"keywords": response_content_2}), 200

    except Exception as e:
        print(f"An error occurred: {e}")  # Debugging statement
        traceback.print_exc()  # Print the full traceback to the console
        return jsonify({"error": str(e)}), 500

@app.route('/clear_history', methods=['GET', 'POST'])
def clear_history():
    # Clear the conversation history
    session['conversation_history'] = []
    session['user_questions'] = []
    session.modified = True
    print("History cleared:", session['conversation_history']) 
    print("History cleared:", session['user_questions']) 
    return jsonify({"status": "Conversation history cleared."}), 200


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, port=5001)

