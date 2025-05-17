from langgraph.graph import StateGraph, START, END
from typing import List, Literal, TypedDict
from openai import OpenAI
from pydantic import BaseModel
import dotenv
import os
import json
dotenv.load_dotenv(override=True)

secret = os.getenv("API_KEY")
client = OpenAI(api_key=secret)

# Response models
class CheckQueryResponse(BaseModel):
    is_valid: bool

class BmiResponse(BaseModel):
    BMI: int

class TdeeResponse(BaseModel):
    TDEE: int
    comment_on_cut_or_bulk: str

class BulkCutResponse(BaseModel):
    Bulk: bool
    cut: bool

class FinalResponse(BaseModel):
    finalWordit: str


class State(TypedDict):
    user_msgs: str
    BMI: int
    TDEE: int
    Cut: bool
    bulk: bool
    is_valid:bool
    comment_on_cut_or_bulk: str

def Get_bmi_input(state: State):
    print("Please provide your height, weight, age, and gender")
    check = input("Write: ")
    user_msgs = check
    bmi_check_prompt = "You need height, weight, age, and gender Return False if any is missing, else True."
    check = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        response_format=CheckQueryResponse,
        messages=[
            {"role": "system", "content": bmi_check_prompt},
            {"role": "user", "content": user_msgs}
        ]
    )
    state["is_valid"] = check.choices[0].message.parsed.is_valid
    state["user_msgs"]= user_msgs
    return state

def Get_tdee_input(state: State) :
    print("Please tell me your activity level (low, moderate, high, athlete level).")
    check = input("Write: ")
    user_msgs = check
    tdee_check_prompt = "Check if activity level is provided. Return False if not. Then calculate TDEE and suggest cut or bulk."
    check = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        response_format=CheckQueryResponse,
        messages=[
            {"role": "system", "content": tdee_check_prompt},
            {"role": "user", "content": user_msgs}
        ]
    )
    state["is_valid"] = check.choices[0].message.parsed.is_valid
    state["user_msgs"]= user_msgs
    return state

def Get_cut_bulk_input(state: State):
    print("Do you want to cut or bulk?")
    check = input("Write: ")
    user_msgs = check
    cut_bulk_check_prompt = "User needs to say 'cut' or 'bulk'. Return False if missing."
    check = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        response_format=CheckQueryResponse,
        messages=[
            {"role": "system", "content": cut_bulk_check_prompt},
            {"role": "user", "content": user_msgs}
        ]
    )
    state["is_valid"] = check.choices[0].message.parsed.is_valid
    state["user_msgs"]= user_msgs
    return state

def BMI_routing(state:State)->Literal["Get_bmi_input","calculate_bmi"]:
    isvalid = state.get("is_valid")
    if isvalid:
        return "calculate_bmi"
    else:
        print("cant find height,weight,age or gender")
        return "Get_bmi_input"

def TDEE_routing(state:State)->Literal["Get_tdee_input","calculate_tdee"]:
    isvalid = state.get("is_valid")
    if isvalid:
        return "calculate_tdee"
    else:
        print("cant find activity level")
        return "Get_tdee_input"


def CUT_BULK_routing(state:State)->Literal["Get_cut_bulk_input","choose_cut_bulk"]:
    isvalid = state.get("is_valid")
    if isvalid:
        return "choose_cut_bulk"
    else:
        print("cant find 'cut' or 'bulk'")
        return "Get_cut_bulk_input"

def calculate_bmi(state: State):
    bmi_res = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        response_format=BmiResponse,
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": "Calculate BMI from user data."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": state["user_msgs"]}]
            }
        ]
    )
    state["BMI"] = bmi_res.choices[0].message.parsed.BMI
    print(f"Your BMI is {state['BMI']}. Now, please tell me your activity level.")
    return state

def calculate_tdee(state: State):
    print("Calculating TDEE...")
    tdee_res = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        response_format=TdeeResponse,
        messages=[
            {"role": "system", "content": "Calculate TDEE from user activity level."},
            {"role": "user", "content": state["user_msgs"]}
        ]
    )
    state["TDEE"] = tdee_res.choices[0].message.parsed.TDEE
    state["comment_on_cut_or_bulk"] = tdee_res.choices[0].message.parsed.comment_on_cut_or_bulk
    print(f"Your TDEE is {state["TDEE"]}. Recommendation: {state["comment_on_cut_or_bulk"]}")
    return state


def choose_cut_bulk(state: State):
    res = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        response_format=BulkCutResponse,
        messages=[
            {"role": "system", "content": "Detect if user wants to cut or bulk."},
            {"role": "user", "content": state["user_msgs"]}
        ]
    )
    state["bulk"] = res.choices[0].message.parsed.Bulk
    state["Cut"] = res.choices[0].message.parsed.cut
    return state

def final_plan(state: State):
    prompt = """
    Generate a final diet plan based on BMI, TDEE, and user's choice (cut or bulk).
    Provide 3 options: Normal, Moderately Aggressive, and Aggressive.
    """
    res = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        response_format=FinalResponse,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(state)}
        ]
    )
    print(res.choices[0].message.parsed.finalWordit)
    return state

graph_builder = StateGraph(State)

graph_builder.add_node("Get_bmi_input", Get_bmi_input)
# graph_builder.add_node("route_bmi",BMI_routing)
graph_builder.add_node("calculate_bmi", calculate_bmi)
graph_builder.add_node("Get_tdee_input", Get_tdee_input)
# graph_builder.add_node("route_tdee",TDEE_routing)
graph_builder.add_node("calculate_tdee", calculate_tdee)
graph_builder.add_node("Get_cut_bulk_input", Get_cut_bulk_input)
# graph_builder.add_node("route_cut_bulk",CUT_BULK_routing)
graph_builder.add_node("choose_cut_bulk", choose_cut_bulk)
graph_builder.add_node("final_plan", final_plan)

graph_builder.add_edge(START, "Get_bmi_input")
# graph_builder.add_edge("Get_bmi_input", "route_bmi")
graph_builder.add_conditional_edges("Get_bmi_input", BMI_routing)
graph_builder.add_edge("calculate_bmi", "Get_tdee_input")
graph_builder.add_conditional_edges("Get_tdee_input", TDEE_routing)
graph_builder.add_edge("calculate_tdee", "Get_cut_bulk_input")
graph_builder.add_conditional_edges("Get_cut_bulk_input", CUT_BULK_routing)
graph_builder.add_edge("choose_cut_bulk","final_plan")
graph_builder.add_edge("final_plan", END)

graph = graph_builder.compile()


def call_graph():
    state = State(user_msgs=[], is_valid=True, BMI=0, TDEE=0, bulk=False, Cut=False, comment_on_cut_or_bulk="")
    result = graph.invoke(state)
    print("Final State:", result)

call_graph()