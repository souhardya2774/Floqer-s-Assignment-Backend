from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)
from llama_index.experimental.query_engine.pandas import (
    PandasInstructionParser,
)
from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

origins = [
    "https://floqer-s-assignment-pi.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    query: str

df = pd.read_csv("salaries.csv")

instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
)

context_str= (
    "This dataset represents salary information for machine learning engineers in years 2020 to 2024.\n"
    "It includes details such as the level of experience, employment type, job title, salary amount, currency, employee residence, remote work ratio, company location, and company size.\n"
    "This information allows for analysis of salary trends, employment patterns, and other factors affecting machine learning engineer salaries in various locations and company settings.\n"
    "Description of the features in dataset:\n"
    "1. work_year: The year in which the salary data was collected (e.g., 2024 or 2023).\n"
    "2. experience_level: The level of experience of the employee (e.g., MI for Mid-Level).\n"
    "3. employment_type: The type of employment (e.g., FT for Full-Time).\n"
    "4. job_title: The title of the job (e.g., Data Scientist).\n"
    "5. salary: The salary amount.\n"
    "6. salary_currency: The currency in which the salary is denominated (e.g., USD for US Dollars).\n"
    "7. salary_in_usd: The salary amount converted to US Dollars.\n"
    "8. employee_residence: The country of residence of the employee (e.g., AU for Australia).\n"
    "9. remote_ratio: The ratio indicating the level of remote work (0 for no remote work).\n"
    "10. company_location: The location of the company (only country) (e.g., AU for Australia).\n"
    "11. company_size: The size of the company (not compant name) (e.g., S for Small).\n"
)

pandas_prompt_str = (
    "You are a Data Scientist, working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Context of the dataframe:\n"
    "{context}"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)

response_synthesis_prompt_str = (
    "You are a Data analyst.\n"
    "Given an input question, synthesize a response from the query results and give a understandable answer.\n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"
    "Caution: If no results there just causually ask to give discuss question or greed them if they are greeting you.\n"
    "Response: "
)


pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str= instruction_str, df_str= df.head(5), context= context_str
)
pandas_output_parser = PandasInstructionParser(df)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
llm = Groq(model="llama-3.1-8b-instant")

print(pandas_output_parser)

qp = QP(
    modules={
        "input": InputComponent(),
        "pandas_prompt": pandas_prompt,
        "llm1": llm,
        "pandas_output_parser": pandas_output_parser,
        "response_synthesis_prompt": response_synthesis_prompt,
        "llm2": llm,
    },
    verbose=True,
)
qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
qp.add_links(
    [
        Link("input", "response_synthesis_prompt", dest_key="query_str"),
        Link(
            "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
        ),
        Link(
            "pandas_output_parser",
            "response_synthesis_prompt",
            dest_key="pandas_output",
        ),
    ]
)
# add link from response synthesis prompt to llm2
qp.add_link("response_synthesis_prompt", "llm2")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/query")
async def query(item:Item):
    print(item)
    response = qp.run(
        query_str=item.query
    )

    return {"message":response.message.content}