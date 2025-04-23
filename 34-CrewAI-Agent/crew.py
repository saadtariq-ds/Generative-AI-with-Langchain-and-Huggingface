import os
from crewai import Crew, Process
from agents import blog_researcher, blog_writer
from tasks import research_task,writer_task
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


crew = Crew(
    agents=[blog_researcher, blog_writer],
    tasks=[research_task, writer_task],
    process=Process.sequential,
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True,
)

## Start the Execution
result = crew.kickoff(
    inputs={'topic':'AI vs ML vs DL vs Data Science'}
)
print(result)