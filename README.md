# Artificial Intelligence

A collection of Python-based AI engineering focused on assistant orchestration, function calling, tool execution patterns, document analysis, and practical AI system design.

This repository is organized as a portfolio-friendly AI engineering workspace. It includes a modernized assistant implementation under `src/` and supporting examples under `patterns/`.

## Project Goals

This project demonstrates practical AI engineering patterns, including:

- OpenAI assistant lifecycle management
- Assistant creation, update, retrieval, and deletion
- Thread-based conversation handling
- Streaming assistant responses
- Function and tool-calling workflows
- Safe local tool execution fallback
- Chat completion patterns
- Document analysis workflows
- AI systems engineering assistant design
- Separation between core assistant logic and reusable implementation patterns

## Repository Structure

```text
artificial-intelligence/
├── src/
│   ├── ai_assistant.py
│   ├── tool_functions.json
│   └── openai-assistants.json
├── patterns/
│   ├── assistant/
│   ├── chat_completion/
│   ├── airtable_fc.py
│   ├── code_interpreter_wout_stream.py
│   ├── function_calling_with_stream.py
│   └── function_calling_wout_stream.py
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

The repository is organized into two main areas:

- `src/` contains the primary assistant implementation.
- `patterns/` contains reusable AI engineering patterns, including assistant orchestration, function calling, document analysis, tool execution, and chat completion workflows.

## Main Implementation

The primary assistant implementation is located in:

```text
src/ai_assistant.py
```

This file demonstrates a structured assistant lifecycle, including assistant creation, update, retrieval, thread creation, streaming responses, and tool-call handling.

The assistant is designed as an **AI Systems Engineering Assistant** focused on:

- RAG pipeline design
- Agentic workflow design
- Tool and function orchestration
- Grounded generation
- Answer verification
- AI safety guardrails
- Evaluation frameworks
- Observability and tracing
- Reliability, scalability, and maintainability

## Patterns Directory

The `patterns/` directory contains reusable AI engineering patterns and implementation references.

It includes:

- Assistant orchestration patterns
- Chat completion workflows
- Function-calling examples
- Streaming and non-streaming execution patterns
- Code interpreter usage patterns
- Document analysis examples
- Tool execution examples
- Domain-specific workflow samples such as financial, flight, stock, weather, and home automation-style handlers

The goal of this directory is to preserve reusable implementation approaches while keeping the main assistant implementation clean and easy to review.

## Requirements

- Python 3.12+
- OpenAI Python SDK
- A valid OpenAI API key

## Environment Setup

Create a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local environment file:

```bash
cp .env.example .env
```

Add your OpenAI API key to `.env`:

```env
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_ASSISTANT_MODEL=gpt-4o
```

## Running the Assistant

From the project root:

```bash
cd src
python ai_assistant.py
```

Example prompts:

```text
Design a grounded RAG architecture for a large document corpus.
Review this AI agent architecture for reliability and observability.
Create an evaluation plan for a RAG assistant that must avoid unsupported answers.
```

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
