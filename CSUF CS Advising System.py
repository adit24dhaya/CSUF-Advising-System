import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple


logging.basicConfig(
    level=logging.INFO,
    filename="advising_system.log",
    filemode="a",
    format="%(asctime)s - %(message)s",
)

CSUF_LINKS: Dict[str, str] = {
    "csuf": "https://www.fullerton.edu/ecs/cs/",
    "advising": "https://www.fullerton.edu/ecs/cs/resources/advisement.php",
    "graduation": "https://www.fullerton.edu/ecs/cs/resources/graduation.php",
    "prerequisites": "https://www.fullerton.edu/ecs/cs/_resources/pdf/course_plan/BS-CS%20Prerequisite%20Relation_v.2024-04-11.pdf",
}

CSUF_COURSES: Dict[str, str] = {
    "CPSC 120A": "Intro to Programming Lecture",
    "CPSC 120L": "Intro to Programming Lab",
    "CPSC 121A": "Object-Oriented Programming Lecture",
    "CPSC 121L": "Object-Oriented Programming Lab",
    "CPSC 131": "Data Structure",
    "CPSC 223X": "Programming in C/Java/C#/Python/Swift",
    "CPSC 240": "Computer Organization and Assembly Language",
    "CPSC 253": "Cybersecurity Foundations and Principles",
    "CPSC 254": "Software Development with Open Source Systems",
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
    "EGGN 495": "Professional Practice (Internship)",
}

MATH_REQUIREMENTS: Dict[str, str] = {
    "MATH 150A": "Calculus 1",
    "MATH 150B": "Calculus 2",
    "MATH 170A": "Math Structures 1",
    "MATH 170B": "Math Structures 2",
    "STATISTICS": "Applied to Natural Sciences",
}

COURSE_UNITS: Dict[str, int] = {
    "CPSC 120L": 1,
    "CPSC 121L": 1,
}
DEFAULT_UNITS = 3

PREREQUISITES: Dict[str, List[str]] = {
    "CPSC 121A": ["CPSC 120A"],
    "CPSC 121L": ["CPSC 120L"],
    "CPSC 131": ["CPSC 121A", "CPSC 121L"],
    "CPSC 240": ["CPSC 131"],
    "CPSC 335": ["CPSC 131", "MATH 170A"],
    "CPSC 351": ["CPSC 131", "CPSC 240"],
    "CPSC 362": ["CPSC 131"],
    "CPSC 483": ["CPSC 131", "MATH 150A", "STATISTICS"],
    "CPSC 452": ["CPSC 131", "CPSC 240"],
}

ENHANCED_TEMPLATE = """You are an AI academic advisor for the Computer Science department at California State University, Fullerton (CSUF).

### Context:
{context}

### Student Question:
{question}

### Response:
"""


@dataclass
class AdvisingResult:
    response: str
    mode: str
    completed_courses: List[str]


def normalize_course_code(raw: str) -> str:
    text = re.sub(r"\s+", " ", raw.upper()).strip()
    text = text.replace("-", " ")
    return text


def parse_completed_courses(text: Optional[str]) -> List[str]:
    if not text:
        return []
    items = [normalize_course_code(token) for token in text.split(",")]
    return [item for item in items if item]


def extract_completed_from_query(query: str) -> Tuple[str, List[str]]:
    """
    Extracts a suffix like:
      completed: CPSC 120A, CPSC 120L, MATH 150A
    """
    pattern = re.compile(r"completed\s*:\s*(.+)$", re.IGNORECASE)
    match = pattern.search(query)
    if not match:
        return query.strip(), []
    courses = parse_completed_courses(match.group(1))
    clean_query = pattern.sub("", query).strip(" ,")
    return clean_query, courses


def estimate_units(courses: Sequence[str]) -> int:
    return sum(COURSE_UNITS.get(c, DEFAULT_UNITS) for c in courses)


def find_eligible_next_courses(completed: Set[str]) -> List[str]:
    eligible = []
    for course, prereqs in PREREQUISITES.items():
        if course in completed:
            continue
        if all(pr in completed for pr in prereqs):
            eligible.append(course)
    return sorted(eligible)


def build_progress_summary(completed_courses: Sequence[str]) -> List[str]:
    if not completed_courses:
        return []
    completed_set = set(completed_courses)
    completed_units = estimate_units(completed_courses)
    remaining_units = max(0, 120 - completed_units)
    eligible = find_eligible_next_courses(completed_set)
    summary = [
        f"Progress estimate: {completed_units} completed units, ~{remaining_units} units remaining to 120.",
        f"Completed courses tracked ({len(completed_courses)}): {', '.join(completed_courses[:12])}"
        + (" ..." if len(completed_courses) > 12 else ""),
    ]
    if eligible:
        summary.append(f"Likely eligible next courses: {', '.join(eligible[:8])}")
    else:
        summary.append("No clear next-course recommendations detected from current prerequisite map.")
    return summary


def build_fast_advising_response(query: str, completed_courses: Optional[List[str]] = None) -> str:
    q = query.lower()
    lines: List[str] = []
    completed_courses = completed_courses or []

    if any(k in q for k in ["graduation", "graduate", "units", "progress", "what-if"]):
        lines.append("Graduation baseline: 120 total units, including 39 upper-division units.")
        lines.append("You should also satisfy CS core, GE, and science/math elective requirements.")

    if "machine learning" in q or "cpsc 483" in q:
        lines.append("CPSC 483 prerequisites: CPSC 131, MATH 150A, and Statistics.")
    if "cryptography" in q or "cpsc 452" in q:
        lines.append("CPSC 452 prerequisites: CPSC 131 and CPSC 240.")

    matched_courses = [course for course in CSUF_COURSES if course.lower() in q]
    if matched_courses:
        for course in matched_courses[:4]:
            lines.append(f"{course}: {CSUF_COURSES[course]}")

    if any(k in q for k in ["math", "calculus", "statistics"]):
        lines.append("Math requirements include: MATH 150A, MATH 150B, MATH 170A, MATH 170B, and Statistics.")

    if completed_courses:
        lines.extend(build_progress_summary(completed_courses))

    if any(k in q for k in ["next course", "take next", "eligible next", "schedule"]) and not completed_courses:
        lines.append("Tip: add your completed list like `completed: CPSC 120A, CPSC 120L, CPSC 121A, CPSC 121L`.")

    relevant_links = [f"- {name.capitalize()}: {url}" for name, url in CSUF_LINKS.items() if name in q]
    if relevant_links:
        lines.append("Useful links:")
        lines.extend(relevant_links)

    if not lines:
        lines.append("I can help with CSUF CS prerequisites, graduation progress, and next-course recommendations.")
        lines.append("Try: `What can I take next? completed: CPSC 120A, CPSC 120L, CPSC 121A, CPSC 121L`")

    return "\n".join(lines)


def load_base_context() -> str:
    return (
        "At CSUF, prerequisites typically include a grade of C or better in foundational courses. "
        "Graduation generally requires 120 total units and 39 upper-division units."
    )


def load_quantized_model():
    from huggingface_hub import login
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    model_name = "meta-llama/Llama-2-7b-chat-hf"
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    if not huggingface_token:
        raise RuntimeError("Missing HUGGINGFACE_TOKEN environment variable.")

    login(token=huggingface_token)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        load_in_4bit_use_double_quant=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=huggingface_token,
    )
    return model, tokenizer, device


def generate_response(model, tokenizer, base_context: str, query: str, links_dict: Dict[str, str], device: str, max_tokens: int = 280) -> str:
    try:
        context = base_context
        if "machine learning" in query.lower() or "cpsc 483" in query.lower():
            context += "\nCPSC 483 prerequisites: CPSC 131, MATH 150A, Statistics."
        if "cryptography" in query.lower() or "cpsc 452" in query.lower():
            context += "\nCPSC 452 prerequisites: CPSC 131, CPSC 240."

        combined_input = ENHANCED_TEMPLATE.format(context=context, question=query)
        inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, max_length=1024).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
        )
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = raw_output.split("### Response:")[-1].strip()
        links = [f"- [{key.capitalize()}]({url})" for key, url in links_dict.items() if key in query.lower()]
        if links:
            answer += f"\n\nUseful links:\n{chr(10).join(links)}"
        return answer
    except Exception as exc:
        logging.error("Error generating LLM response: %s", exc)
        return "LLM response failed. Try fast mode or a simpler query."


def answer_query(query: str, mode: str, model=None, tokenizer=None, device: str = "cpu", completed_from_args: Optional[List[str]] = None) -> AdvisingResult:
    parsed_query, completed_from_query = extract_completed_from_query(query)
    merged_completed = sorted(set((completed_from_args or []) + completed_from_query))

    if mode == "llm" and model is not None and tokenizer is not None:
        base_context = load_base_context()
        response = generate_response(model, tokenizer, base_context, parsed_query, CSUF_LINKS, device)
        if merged_completed:
            response += "\n\n" + "\n".join(build_progress_summary(merged_completed))
        return AdvisingResult(response=response, mode="llm", completed_courses=merged_completed)

    response = build_fast_advising_response(parsed_query, merged_completed)
    return AdvisingResult(response=response, mode="fast", completed_courses=merged_completed)


def parse_args():
    parser = argparse.ArgumentParser(description="CSUF Advising System")
    parser.add_argument("--mode", choices=["fast", "llm"], default="fast", help="Use fast rule-based advising or quantized LLM mode.")
    parser.add_argument("--query", help="Optional one-shot question. If omitted, runs interactive mode.")
    parser.add_argument("--completed-courses", default="", help="Comma-separated completed courses for progress analysis.")
    parser.add_argument("--json-output", help="Optional path to save response payload as JSON.")
    return parser.parse_args()


def maybe_write_json(path: Optional[str], payload: Dict[str, object]) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_interactive(mode: str, model=None, tokenizer=None, device: str = "cpu", completed_from_args: Optional[List[str]] = None):
    print("Welcome to the CS Advising System!")
    print("Type 'exit' to quit.\n")
    while True:
        try:
            query = input("Your question: ").strip()
            if query.lower() == "exit":
                print("Goodbye!")
                break
            result = answer_query(query, mode=mode, model=model, tokenizer=tokenizer, device=device, completed_from_args=completed_from_args)
            print(f"\nResponse ({result.mode} mode):\n{result.response}\n")
        except KeyboardInterrupt:
            print("\nSession ended by user.")
            break
        except Exception as exc:
            print(f"An error occurred: {exc}")
            logging.error("Interactive loop error: %s", exc)


def main():
    args = parse_args()
    mode = args.mode
    model = None
    tokenizer = None
    device = "cpu"
    completed_from_args = parse_completed_courses(args.completed_courses)

    if mode == "llm":
        try:
            model, tokenizer, device = load_quantized_model()
            print("Quantized model loaded successfully!")
        except Exception as exc:
            print(f"LLM mode unavailable ({exc}). Falling back to fast mode.")
            logging.error("LLM mode fallback triggered: %s", exc)
            mode = "fast"

    if args.query:
        result = answer_query(
            args.query,
            mode=mode,
            model=model,
            tokenizer=tokenizer,
            device=device,
            completed_from_args=completed_from_args,
        )
        payload = {
            "mode": result.mode,
            "query": args.query,
            "completed_courses": result.completed_courses,
            "response": result.response,
        }
        maybe_write_json(args.json_output, payload)
        print(result.response)
        return

    run_interactive(mode=mode, model=model, tokenizer=tokenizer, device=device, completed_from_args=completed_from_args)


if __name__ == "__main__":
    main()

