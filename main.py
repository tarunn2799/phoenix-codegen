import argparse
from logic_gen.logic_flow import generate_logic

def main():
    parser = argparse.ArgumentParser(description="Generate logic based on task description")
    parser.add_argument("--task", type=str, default="I want to extract pii entities from text",
                        help="Description of the task to generate logic for")
    args = parser.parse_args()
    
    logic_file = generate_logic(args.task)
    print(f"Logic generated and saved to: {logic_file}")

if __name__ == "__main__":
    main()