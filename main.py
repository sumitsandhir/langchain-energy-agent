import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
import os

# Load energy plans data
ENERGY_PLANS_CSV = "data/energy_plans.csv"
plans_df = pd.read_csv(ENERGY_PLANS_CSV)

# Define a custom tool
@tool
def get_best_plan(usage_kwh: int) -> str:
    """
    Given the user's energy usage in kWh, return the best matching plan based on cost.
    """
    plans_df['estimated_cost'] = plans_df['rate_per_kwh'] * usage_kwh
    best_plan = plans_df.sort_values(by='estimated_cost').iloc[0]
    return f"Best Plan: {best_plan['name']}, Estimated Monthly Cost: £{best_plan['estimated_cost']:.2f}"

# Set up LangChain agent
llm = ChatOpenAI(temperature=0, model="gpt-4")
tools = [get_best_plan]
agent = initialize_agent(tools, llm, agent="chat-zero-shot-react-description", verbose=True)

# FastAPI app
app = FastAPI()

class UserInput(BaseModel):
    prompt: str

@app.post("/query")
async def query_agent(user_input: UserInput):
    response = agent.run(user_input.prompt)
    return {"response": response}
