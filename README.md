# CSUF-Advising-System
This project is a conversational AI-powered advising system designed to answer student questions about the Computer Science program at California State University, Fullerton (CSUF). It leverages the power of Large Language Models (LLMs) and efficient quantization techniques to provide helpful and informative responses.

## Description

Navigating university requirements and course selection can be challenging for students. This advising system aims to simplify the process by providing instant access to information about:

*   Course prerequisites
*   Graduation requirements
*   Relevant CSUF resources and links
*   General advising questions

The system utilizes a pre-trained Large Language Model (specifically, a quantized version of Llama 2), fine-tuned with information specific to the CSUF CS department. This allows it to understand natural language queries and provide contextually relevant answers.

## Key Features

*   **Conversational Interface:** Users can ask questions in natural language.
*   **Contextual Awareness:** The system considers the context of the conversation and provides relevant information.
*   **CSUF Specific Knowledge:** The model is trained on CSUF CS department information, including course details, prerequisites, and graduation requirements.
*   **Link Integration:** Relevant CSUF website links are provided within the responses for further exploration.
*   **Efficient Inference:** 4-bit quantization using the `bitsandbytes` library enables efficient use of resources.

## Technologies Used

*   Python
*   Transformers (Hugging Face)
*   bitsandbytes
*   PyTorch
*   Hugging Face Hub

## Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/](https://github.com/)[YourUsername]/[YourRepositoryName].git
    ```

2.  Navigate to the project directory:

    ```bash
    cd [YourRepositoryName]
    ```

3.  Install the required packages:

    ```bash
    pip install bitsandbytes transformers accelerate --upgrade
    ```

4.  Log in to your Hugging Face account (You will need a Hugging Face account and an access token):

    ```python
    from huggingface_hub import login
    huggingface_token = "YOUR_HUGGING_FACE_TOKEN" # Replace with your token
    login(token=huggingface_token)
    ```

## Usage

Run the `advising_system.py` script:

```bash
python advising_system.py
