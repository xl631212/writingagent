import streamlit as st
import warnings
import os
from crewai import Agent, Task, Crew

# Suppress warnings
warnings.filterwarnings('ignore')

# Streamlit App Title
st.title("Interactive CrewAI Content Creation")

# OpenAI API Key Input
openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")

if openai_api_key:
    os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # User Input for Topic
    topic = st.text_input("Enter the topic for content creation:", value="Artificial Intelligence")

    if topic:
        # Verbosity Checkbox
        verbose = st.checkbox("Show detailed debug logs", value=True)

        # Define Agents
        planner = Agent(
            role="Content Planner",
            goal=f"Plan engaging and factually accurate content on {topic}",
            backstory=(
                f"You're planning a blog article about the topic: {topic}. "
                "You collect information to help the audience learn something "
                "and make informed decisions."
            ),
            allow_delegation=False,
            verbose=verbose
        )
        writer = Agent(
            role="Content Writer",
            goal=f"Write insightful and factually accurate content about {topic}",
            backstory=(
                f"You're crafting a blog article about the topic: {topic}, "
                "using the outline from the Content Planner."
            ),
            allow_delegation=False,
            verbose=verbose
        )
        editor = Agent(
            role="Editor",
            goal="Edit the blog post to ensure quality and alignment with the organization's style.",
            backstory=(
                "You are editing a blog post created by the Content Writer, "
                "ensuring it adheres to journalistic practices and avoids controversies."
            ),
            allow_delegation=False,
            verbose=verbose
        )

        # Define Tasks
        plan = Task(
            description=(
                f"1. Prioritize trends, key players, and noteworthy news about {topic}.\n"
                "2. Identify the target audience and their interests.\n"
                "3. Develop a detailed content outline, including SEO keywords and resources."
            ),
            expected_output="A comprehensive content plan document.",
            agent=planner
        )
        write = Task(
            description=(
                f"1. Use the content plan to write a compelling blog post on {topic}.\n"
                "2. Incorporate SEO keywords naturally and structure the post properly.\n"
                "3. Ensure the blog has an engaging introduction, body, and conclusion."
            ),
            expected_output="A well-written blog post in markdown format.",
            agent=writer
        )
        edit = Task(
            description="Proofread the blog post for grammatical errors and style alignment.",
            expected_output="A polished blog post in markdown format.",
            agent=editor
        )

        # Assemble Crew
        crew = Crew(
            agents=[planner, writer, editor],
            tasks=[plan, write, edit],
            verbose=2 if verbose else 0
        )

        # Start Button
        if st.button("Generate Content"):
            st.info("Processing... This may take a few moments.")
            with st.spinner("The agents are working on your content..."):
                try:
                    result = crew.kickoff(inputs={"topic": topic})
                    st.success("Content generation completed!")

                    # Display Outputs
                    st.subheader("Planner Output")
                    st.text(result["plan"]["output"])

                    st.subheader("Writer Output")
                    st.text(result["write"]["output"])

                    st.subheader("Editor Output")
                    st.text(result["edit"]["output"])

                    # Debug Logs (Optional)
                    if verbose:
                        st.subheader("Debug Logs")
                        st.text(result)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
else:
    st.warning("Please provide your OpenAI API key to proceed.")
