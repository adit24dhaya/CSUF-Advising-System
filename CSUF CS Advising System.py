pip install bitsandbytes transformers accelerate --upgrade

import logging
import torch
import itertools
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

# Authenticate with HuggingFace
huggingface_token = "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"  # Replace with your token
login(token=huggingface_token)

# Configure logging
logging.basicConfig(level=logging.INFO, filename="advising_system.log", filemode="a",
                    format="%(asctime)s - %(message)s")

# Load the quantized model and tokenizer
def load_quantized_model():
    model_name = "meta-llama/Llama-2-7b-chat-hf"

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        load_in_4bit_use_double_quant=True
    )

    # Detect and set device (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Load model with quantization and assign device
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically uses GPU if available
        use_auth_token=True
    )

    return model, tokenizer, device

# Initialize the quantized model and tokenizer
model, tokenizer, device = load_quantized_model()
print("Quantized model loaded successfully!")

# CSUF links dictionary
csuf_links = {
    "csuf": "https://www.fullerton.edu/ecs/cs/",
    "advising": "https://www.fullerton.edu/ecs/cs/resources/advisement.php",
    "graduation": "https://www.fullerton.edu/ecs/cs/resources/graduation.php",
    "prerequisites": "https://www.fullerton.edu/ecs/cs/_resources/pdf/course_plan/BS-CS%20Prerequisite%20Relation_v.2024-04-11.pdf",
}

# Context Template
enhanced_template = """You are an AI academic advisor for the Computer Science department at California State University, Fullerton (CSUF).

### Context:
{context}

### Student Question:
{question}

### Response:
"""

# Course Listings
csuf_courses = {
    "CPSC 120A": "Intro to Programming Lecture",
    "CPSC 120L": "Intro to Programming Lab",
    "CPSC 121A": "Object-Oriented Programming Lecture",
    "CPSC 121L": "Object-Oriented Programming Lab",
    "CPSC 131": "Data Structure",
    "CPSC 223x": "Programming in C/Java/C#/Python/Swift",
    "CPSC 240": "Computer Organization and Assembly Language",
    "CPSC 253": "Cybersecurity Foundations and Principles",
    "CPSC 254": "Software Development with Open Source Systems",
    "CPSC 254*new": "Applied Artificial Intelligence",
    "CPSC 315": "Professional Ethics in Computing",
    "CPSC 323": "Compilers and Languages",
    "CPSC 332": "File Structures & Database Systems",
    "CPSC 335": "Algorithm Engineering",
    "CPSC 351": "Operating Systems Concepts",
    "CPSC 362": "Foundations of Software Engineering",
    "CPSC 375": "Intro to Data Science and Big Data",
    "CPSC 386": "Intro to Game Design and Production",
    "CPSC 411": "Mobile Device Application Programming (iOS)",
    "CPSC 411A": "Mobile Device App Programming for Android",
    "CPSC 431": "Database and Applications",
    "CPSC 439": "Theory of Computation",
    "CPSC 440": "Computer System Architecture",
    "CPSC 449": "Web Back-End Engineering",
    "CPSC 452": "Cryptography",
    "CPSC 454": "Cloud Computing and Security",
    "CPSC 455": "Web Security",
    "CPSC 456": "Network Security Fundamentals",
    "CPSC 458": "Malware Analysis",
    "CPSC 459": "Blockchain Technologies",
    "CPSC 462": "Software Design",
    "CPSC 463": "Software Testing",
    "CPSC 464": "Software Architecture",
    "CPSC 466": "Software Process",
    "CPSC 471": "Computer Communications",
    "CPSC 474": "Parallel & Distributed Computing",
    "CPSC 479": "Intro to High Performance Computing",
    "CPSC 481": "Artificial Intelligence",
    "CPSC 483": "Intro to Machine Learning",
    "CPSC 484": "Principles of Computer Graphics",
    "CPSC 485": "Computational Bioinformatics",
    "CPSC 486": "Game Programming",
    "CPSC 487": "Computational Epidemiology",
    "CPSC 488": "Natural Language Processing",
    "CPSC 490": "Undergraduate Seminar in CS",
    "CPSC 491": "Senior Capstone Project in CS",
    "CPSC 499": "Independent Study",
    "EGGN 495": "Professional Practice (Internship)"
}

# Math Requirements
math_requirements = {
    "Math 150A": "Calculus 1",
    "Math 150B": "Calculus 2",
    "Math 170A": "Math Structures 1",
    "Math 170B": "Math Structures 2",
    "Statistics": "Applied to Natural Sciences"
}

# Categories
categories = {
    "Core Requirements": "Upper Division Core (30 units)",
    "General Education": "GE (24 units)",
    "Graduation Requirement": "(3 units)",
    "Science/Math Electives": "(12 units)"
}

# Base context
def load_base_context():
    return """
    At CSUF, prerequisites include a grade of C or better in foundational programming courses.
    Graduation requires 120 total units, including 39 upper-division units.
    """

# Generate Response
def generate_response(model, tokenizer, base_context, query, links_dict, device, max_tokens=300):
    try:
        # Prepare dynamic context
        context = base_context

        # Inject additional context for specific courses
        course_keywords = {
            "machine learning": {
                "course": "CPSC 483 - Intro to Machine Learning",
                "prerequisites": "CPSC 131 (Data Structures), Math 150A (Calculus 1), and Statistics Applied to Natural Sciences."
            },
            "cryptography": {
                "course": "CPSC 352/452 - Cryptography",
                "prerequisites": "CPSC 131 (Data Structures) and CPSC 240 (Assembly Language)."
            },
        }

        # Match query keywords
        for keyword, details in course_keywords.items():
            if keyword in query.lower():
                context += f"\n\n{details['course']} Prerequisites: {details['prerequisites']}\n"

        # Combine input template
        combined_input = enhanced_template.format(context=context, question=query)
        inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, max_length=1024)

        # Move inputs to the same device as the model
        inputs = inputs.to(device)

        # Generate output
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True
        )
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Format Response
        answer = raw_output.split("### Response:")[-1].strip()

        # Add relevant links
        links = [f"- [{key.capitalize()}]({url})" for key, url in links_dict.items() if key in query.lower()]
        if links:
            answer += f"\n\nUseful links:\n{chr(10).join(links)}"

        return answer
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "An error occurred while generating the response. Please try again."

# Advising System Main Loop
def advising_system():
    print("Welcome to the CS Advising System!")
    print("Type 'exit' to quit.\n")

    base_context = load_base_context()

    while True:
        try:
            query = input("Your question: ").strip()
            if query.lower() == 'exit':
                print("Goodbye!")
                break

            response = generate_response(model, tokenizer, base_context, query, csuf_links, device)
            print(f"\nResponse:\n{response}\n")
        except KeyboardInterrupt:
            print("\nSession ended by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

# Run the advising system
advising_system()

