import os
from crewai import Agent
from tools import youtube_tool
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"]="gpt-4-0125-preview"


## Create a Blog Content Researcher Agent
blog_researcher = Agent(
    role="Blog Creator from Youtube Videos",
    goal="Get the Relevant Video Content for the Topic {topic} from YouTube Channel",
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI Data Science , Machine Learning And GEN AI and providing suggestion"
    ),
    tools=[youtube_tool],
    allow_delegation=True
)


## Create a Blog Writer Agent
blog_writer = Agent(
    role="Blog Writer",
    goal="Narrate Compelling Tech Stories about the Video {topic} from YouTube Channel",
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner."
    ),
    tools=[youtube_tool],
    allow_delegation=False
)