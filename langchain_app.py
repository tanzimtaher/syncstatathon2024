import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
import re
from PIL import Image
import matplotlib.pyplot as plt

# Initialize the Google LLM
api_key = 'AIzaSyAmcSLeR-C21cTsOtyk9eznK2IK52Yu6_c'
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0.2)

def setup_agent(df):
    return create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="zero-shot-react-description",
        verbose=True,
        return_intermediate_steps=True
    )

def process_query(file, query):
    if file is not None:
        df = pd.read_csv(file.name, encoding='ISO-8859-1')
        df = df.head(100)  # Optionally limit the rows
        df = df.iloc[:, :129]
        
        agent_executor = setup_agent(df)
        response = agent_executor.invoke(query, handle_parsing_errors=True)
        #print("LLM Response:", response)  # Add this line to check the actual output

        
        if 'plot' in query.lower() or 'draw' in query.lower():
            try:
                tool_input = response['intermediate_steps'][0][0].tool_input
                # Assuming simple format "plot x vs y"
                x_col = re.search(r"x='([^']+)'", tool_input).group(1)
                y_col = re.search(r"y='([^']+)'", tool_input).group(1)
                kind = re.search(r"kind='([^']+)'", tool_input).group(1)
    
                if x_col in df.columns and y_col in df.columns:
                    df.plot(kind=kind, x=x_col, y=y_col, alpha=0.5)
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    plt.savefig('simple_plot.png')
                    plt.close()
                    img = Image.open('simple_plot.png')
                    return response['output'], df.to_string(), img
            except:
                exec(response["intermediate_steps"][-1][0].tool_input)
                plt.savefig('simple_plot.png')
                plt.close()
                img = Image.open('simple_plot.png')
                return response['output'], df.to_string(), img
            else:
                return response['output'], df.to_string(), "Requested columns not found in dataset"
        return response['output'], df.to_string(), None
    else:
        return "No file uploaded", "No file uploaded", None

iface = gr.Interface(
    fn=process_query,
    inputs=[gr.inputs.File(label="Upload CSV File"), gr.inputs.Textbox(label="Enter Query")],
    outputs=[
        gr.outputs.Textbox(label="Agent Response"),
        gr.outputs.Textbox(label="CSV Preview"),
        gr.outputs.Image(label="Generated Plot", type='pil')  # Specify the type as 'pil'
    ],
    title="LangChain Agent and CSV Viewer",
    description="Upload a CSV file and enter your query to get responses based on the dataset. Include plot commands to see visualizations."
)

iface.launch(debug=True)
