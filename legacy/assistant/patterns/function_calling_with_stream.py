from openai import OpenAI
from openai import AssistantEventHandler
import time
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class EventHandler(AssistantEventHandler):
    def on_event(self, event):
        # print(f"on_event: {event.event}")
        if event.event == 'thread.run.requires_action':
            # print(f"on_event requires_action: {event.event}")
            run_id = event.data.id  # Retrieve the run ID from the event data
            self.handle_requires_action(event.data, run_id)
        elif event.event.startswith('thread.message'):
        # elif event.event == 'thread.message.completed':
        #     print(f"on_event assistant_message: {event.event}")
            # Since it's a complex object, checking for 'content' needs correct path handling
            self.handle_message(event)

    def handle_requires_action(self, data, run_id):
        tool_outputs = []
        for tool in data.required_action.submit_tool_outputs.tool_calls:
            if tool.function.name == "get_current_temperature":
                tool_outputs.append({
                    "tool_call_id": tool.id,
                    "output": "57"  # Simulated temperature value
                })
            elif tool.function.name == "get_rain_probability":
                tool_outputs.append({
                    "tool_call_id": tool.id,
                    "output": "0.06"  # Simulated rain probability
                })

        # Submit all tool outputs at once
        self.submit_tool_outputs(tool_outputs, run_id)

    def submit_tool_outputs(self, tool_outputs, run_id):
        with client.beta.threads.runs.submit_tool_outputs_stream(
                thread_id=self.current_run.thread_id,
                run_id=run_id,
                tool_outputs=tool_outputs,
                event_handler=EventHandler()) as stream:
            for _ in stream.text_deltas:
                pass  # Do not print anything here
                # print(text, end="", flush=True)
        # print("Tool outputs submitted successfully.")

    @staticmethod
    def handle_message(event):
        message = event.data
        # Assuming the content is in the 'data' object and needs to be accessed specifically
        # if message.role == 'assistant':
        if hasattr(message, 'content') and message.content:
            content_block = message.content[0]
            # print(f"AI: {content_block.text.value}")
            print(f"\033[91mAI: {content_block.text.value}\033[0m")
            # if hasattr(content_block, 'text') and hasattr(content_block.text, 'value'):
            #     print(f"Assistant says: {content_block.text.value}")


def send_message(thread_id, content):
    print(f"Sending message to thread {thread_id}...")
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content
    )
    return message


def run_stream(thread_id, assistant_id):
    print(f"Running and streaming on thread {thread_id}...")
    with client.beta.threads.runs.stream(
      thread_id=thread_id,
      assistant_id=assistant_id,
      event_handler=EventHandler()
    ) as stream:
        stream.until_done()


def cancel_run(thread_id, run_id):
    try:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        ).status
        if run_status not in ['completed', 'failed', 'canceled']:
            client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run_id)
            print(f"Run {run_id} canceled successfully.")
        else:
            print(f"Run {run_id} is already {run_status} and cannot be canceled.")
    except Exception as e:
        print(f"Failed to cancel run {run_id}: {e}")


def run_interactive_session():
    assistant_id = 'asst_XXX'
    thread_id = 'thread_XXX'
    '''
    What's the weather in San Francisco today and the likelihood it'll rain?
    What are the atmospheric conditions?
    What's the weather in New York today?
    '''
    print("Starting interactive session. Type 'exit' to stop.")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'stop']:
            print("Exiting interactive session.")
            time.sleep(3)  # Wait for 1 second before polling again
            break
        send_message(thread_id, user_input)
        run_stream(thread_id, assistant_id)


if __name__ == "__main__":
    # cancel_run('thread_XXX', 'run_XXX')
    run_interactive_session()

