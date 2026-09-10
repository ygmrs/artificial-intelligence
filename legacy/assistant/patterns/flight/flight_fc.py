# print out a lot of things to make things clear
import json
import os
import time

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from functions import get_flight_info, book_flight, file_complaint

_ = load_dotenv(find_dotenv())  # read local .env file

api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# step1.A: upload file
file1 = client.files.create(
    file=open("coupons.tsv", "rb"),
    purpose="assistant",
)

# upload this one for the assistant to call the functions
file2 = client.files.create(
    file=open("functions.py", "rb"),
    purpose="assistant",
)

print(file2.id)

file_list = client.files.list()
print(file_list)

# step1.B: define function
get_flight_info_json = {
    "name": "get_flight_info",
    "description": "Get flight information between two locations",
    "parameters": {
        "type": "object",
        "properties": {
            "loc_origin": {
                "type": "string",
                "description": "The departure airport, e.g. DUS",
            },
            "loc_destination": {
                "type": "string",
                "description": "The destination airport, e.g. HAM",
            },
        },
        "required": ["loc_origin", "loc_destination"],
    },
}

book_flight_json = {
    "name": "book_flight",
    "description": "Book a flight based on flight information",
    "parameters": {
        "type": "object",
        "properties": {
            "loc_origin": {
                "type": "string",
                "description": "The departure airport, e.g. DUS",
            },
            "loc_destination": {
                "type": "string",
                "description": "The destination airport, e.g. HAM",
            },
            "datetime": {
                "type": "string",
                "description": "The date and time of the flight, e.g. 2023-01-01 01:01",
            },
            "airline": {
                "type": "string",
                "description": "The service airline, e.g. Lufthansa",
            },
        },
        "required": ["loc_origin", "loc_destination", "datetime", "airline"],
    },
}

file_complaint_json = {
    "name": "file_complaint",
    "description": "File a complaint as a customer",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name of the user, e.g. John Doe",
            },
            "email": {
                "type": "string",
                "description": "The email address of the user, e.g. john@doe.com",
            },
            "text": {
                "type": "string",
                "description": "Description of issue",
            },
        },
        "required": ["name", "email", "text"],
    },
}

available_functions = {
    "get_flight_info": get_flight_info,
    "book_flight": book_flight,
    "file_complaint": file_complaint,
}

# step2: create the assistant with the file
assistant_instruction = (
    "This assistant helps find flight information, book flights, and file complaints."
)

assistant = client.beta.assistants.create(
    name="Test_functional_call",
    instructions=assistant_instruction,
    model=os.getenv("OPENAI_ASSISTANT_MODEL", "gpt-4o"),
    tools=[
        {"type": "retrieval"},
        {"type": "code_interpreter"},
        {"type": "function", "function": get_flight_info_json},
        {"type": "function", "function": book_flight_json},
        {"type": "function", "function": file_complaint_json},
    ],
    file_ids=[file1.id, file2.id],
)

print(assistant.id)

my_assistant = client.beta.assistants.list(
    order="desc",
    limit="20",
)
print(my_assistant.data)

# step3: create a thread
thread = client.beta.threads.create()
print(thread)

# step4: add message to the thread
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="what is the flight information from seattle to austin",
)

# step5: run the assistant to get the response
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)
print(run.id)

# step6: retrieve the run status
print(run.status)

while run.status not in ["completed", "failed", "requires_action"]:
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id,
    )
    print(run.status)
    time.sleep(10)

# step7: handle required tool calls
tool_output_array = []

if run.status == "requires_action":
    tools_to_call = run.required_action.submit_tool_outputs.tool_calls

    for tool in tools_to_call:
        tool_call_id = tool.id
        function_name = tool.function.name
        function_arg = json.loads(tool.function.arguments)

        chosen_function = available_functions.get(function_name)

        if not chosen_function:
            output = json.dumps({"error": f"Unknown function: {function_name}"})
        elif function_name == "get_flight_info":
            output = chosen_function(
                function_arg["loc_origin"],
                function_arg["loc_destination"],
            )
        elif function_name == "book_flight":
            output = chosen_function(
                function_arg["loc_origin"],
                function_arg["loc_destination"],
                function_arg["datetime"],
                function_arg["airline"],
            )
        elif function_name == "file_complaint":
            output = chosen_function(
                function_arg["name"],
                function_arg["email"],
                function_arg["text"],
            )

        tool_output_array.append({
            "tool_call_id": tool_call_id,
            "output": output,
        })

print(tool_output_array)

if tool_output_array:
    run = client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread.id,
        run_id=run.id,
        tool_outputs=tool_output_array,
    )

    print(run.status)

    while run.status not in ["completed", "failed", "requires_action"]:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        print(run.status)
        time.sleep(10)

# step8: display messages
messages = client.beta.threads.messages.list(
    thread_id=thread.id,
)

for m in messages:
    print(m.role + ": " + m.content[0].text.value)
    print("============================")