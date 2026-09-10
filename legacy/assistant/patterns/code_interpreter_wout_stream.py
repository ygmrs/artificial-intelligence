import os
from openai import OpenAI

# Create a global client object
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def create_assistant():
    print("Creating an assistant...")
    assistant = client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor. Write and run code to answer math questions.",
        tools=[{"type": "code_interpreter"}],
        model=os.getenv("OPENAI_ASSISTANT_MODEL", "gpt-4o"),
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
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions="Please address the user as Jane Doe. The user has a premium account."
    )
    print(f"Run status: {run.status}")
    return run


def run():
    assistant = create_assistant()
    thread = create_thread()
    send_message(thread.id, "I need to solve the equation `3x + 11 = 14`. Can you help me?")
    run_result = run_and_poll(thread.id, assistant.id)
    if run_result.status == 'completed':
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        print("Messages:", messages)
        if messages.data:
            # Access the first message and check if it contains 'content'
            first_message = messages.data[0]
            if first_message.content:
                # Extract the text value from the first content block of the first message
                response_text = first_message.content[0].text.value
                print("Extracted Response:", response_text)
            else:
                print("No content or unexpected message in the structure.")
        else:
            print("No messages returned.")
    else:
        print("Run did not complete successfully. Status:", run_result.status)


if __name__ == "__main__":
    run()

