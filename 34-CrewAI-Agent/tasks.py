from crewai import Task
from tools import youtube_tool
from agents import blog_researcher, blog_writer

## Research Task
research_task = Task(
    description=(
        """ Identify the video {topic}.
        Get the detailed information about the video from the channel.
        """
    ),
    expected_output="A comprehensive 3 paragrapsh long report based on the {topic} of the video content",
    tools=[youtube_tool],
    agent=blog_researcher,
)

## Writer Task
writer_task = Task(
    description=(
        "Get the information from the youtube channel on the topic {topic}."
    ),
    expected_output="""Summarize the information from the youtube channel video on the topic {topic}
                    and create content for the blog post""",
    tools=[youtube_tool],
    agent=blog_writer,
    async_execution=False,
    output_file="new_blog_post.md"
)