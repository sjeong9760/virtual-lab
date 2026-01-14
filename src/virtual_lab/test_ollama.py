
import os
import sys
from pathlib import Path

# Add src to python path to ensure imports work if run from root
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from virtual_lab.agent import Agent
    from virtual_lab.run_meeting import run_meeting
except ImportError:
    print("Error: virtual_lab package not found. Please install dependencies or set PYTHONPATH.")
    sys.exit(1)

def test_individual_meeting():
    print("\n--- Testing Individual Meeting (gpt-oss) ---")
    
    researcher = Agent(
        title="Researcher",
        expertise="Molecular Biology",
        goal="Design a new experiment",
        role="Propose a hypothesis",
        model="gpt-oss"
    )
    
    summary = run_meeting(
        meeting_type="individual",
        agenda="Propose a simple experiment to test enzyme activity.",
        save_dir=Path("test_output/individual"),
        save_name="test_meeting",
        team_member=researcher,
        num_rounds=1,
        return_summary=True
    )
    
    print("\nIndividual Meeting Summary:")
    print(summary)

def test_team_meeting():
    print("\n--- Testing Team Meeting (gpt-oss) ---")
    
    lead = Agent(
        title="Lab Head",
        expertise="Computational Biology",
        goal="Oversee the project",
        role="Guide the team",
        model="gpt-oss"
    )
    
    member1 = Agent(
        title="Bioinformatician",
        expertise="Genomics",
        goal="Analyze data",
        role="Process sequences",
        model="gpt-oss"
    )
    
    member2 = Agent(
        title="Chemist",
        expertise="Organic Chemistry",
        goal="Synthesize compounds",
        role="Design synthesis routes",
        model="gpt-oss"
    )
    
    summary = run_meeting(
        meeting_type="team",
        agenda="Discuss a plan to identify novel drug targets for a generic pathogen.",
        save_dir=Path("test_output/team"),
        save_name="team_meeting",
        team_lead=lead,
        team_members=(member1, member2),
        num_rounds=1,
        return_summary=True
    )
    
    print("\nTeam Meeting Summary:")
    print(summary)

if __name__ == "__main__":
    # Create output directories
    os.makedirs("test_output/individual", exist_ok=True)
    os.makedirs("test_output/team", exist_ok=True)
    
    try:
        test_individual_meeting()
        test_team_meeting()
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
