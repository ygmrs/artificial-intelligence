from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = os.getenv("OPENAI_ASSISTANT_MODEL", "gpt-4o"),


def create_assistant():
    print("Creating an assistant...")
    assistant = client.beta.assistants.create(
        name="Weather",
        instructions="You are a weather bot. Use the provided functions to answer questions.",
        model=model,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "description": "Get the current temperature for a specific location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g., San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["Celsius", "Fahrenheit"],
                                "description": "The temperature unit to use. Infer this from the user's location."
                            }
                        },
                        "required": ["location", "unit"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_rain_probability",
                    "description": "Get the probability of rain for a specific location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g., San Francisco, CA"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
    )
    print(f"Assistant Created: {assistant.id}")
    return assistant


def create_thread():
    print("Creating a thread...")
    thread = client.beta.threads.create()
    print(f"Thread Created: {thread.id}")
    return thread


def send_message(thread_id, content):
    print(f"Sending message to thread {thread_id}...")
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content
    )
    print("Message sent.")
    return message


def run_and_poll(thread_id, assistant_id):
    print(f"Running and polling on thread {thread_id}...")
    run_result = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )
    print(f"Run status: {run_result.status}")
    return run_result


def process_run(run, thread_id):
    if run.status == 'completed':
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        print("Messages:", messages)
    else:
        print("Run did not complete successfully. Status:", run.status)


def handle_tool_outputs(run, thread_id):
    tool_outputs = []
    for tool in run.required_action.submit_tool_outputs.tool_calls:
        if tool.function.name == "get_current_temperature":
            tool_outputs.append({
                "tool_call_id": tool.id,
                "output": "57"
            })
        elif tool.function.name == "get_rain_probability":
            tool_outputs.append({
                "tool_call_id": tool.id,
                "output": "0.06"
            })

    if tool_outputs:
        try:
            run = client.beta.threads.runs.submit_tool_outputs_and_poll(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
            print("Tool outputs submitted successfully.")
            process_run(run, thread_id)
        except Exception as e:
            print("Failed to submit tool outputs:", e)


def run():
    assistant_id = 'asst_XXX' # Assistant Created: asst_xxx or create_assistant()
    thread_id = 'thread_XXX' # Thread Created: thread_xxx or create_thread()
    '''
    "What's the weather in San Francisco today and the likelihood it'll rain?"
    "What are the atmospheric conditions?"
    '''
    send_message(thread_id, "What are the atmospheric conditions?")
    run = run_and_poll(thread_id, assistant_id)
    # Retrieve the run status
    print('run status: ', run.status)
    process_run(run, thread_id)

    # Handle tool outputs
    # Check if there are tool outputs to handle
    if hasattr(run, 'required_action') and hasattr(run.required_action, 'submit_tool_outputs'):
        handle_tool_outputs(run, thread_id)


if __name__ == "__main__":
    run()

