import os
import re
import math
import pandas as pd
import plotly.graph_objects as go

path = "log_1775433451"

def generatePlots(file: str, numEpochs=math.inf):
    data = []
    pattern = re.compile(r"Epoch (\d+): Loss -- Train: ([\d.]+) Test: ([\d.]+)    Accuracy -- Train: ([\d.]+) Test: ([\d.]+)")
    
    with open(file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match and int(match.group(1)) <= numEpochs:
                data.append({
                    "Epoch": int(match.group(1)),
                    "Train Loss": float(match.group(2)),
                    "Test Loss": float(match.group(3)),
                    "Train Accuracy": float(match.group(4)),
                    "Test Accuracy": float(match.group(5))
                })
    
    df = pd.DataFrame(data)
    filename = os.path.basename(file)
    
    # Plot loss
    lossPlot = go.Figure()
    lossPlot.add_trace(go.Scatter(x=df["Epoch"], y=df["Train Loss"], name="Train Loss", line=dict(color='#1f77b4', width=2)))
    lossPlot.add_trace(go.Scatter(x=df["Epoch"], y=df["Test Loss"], name="Test Loss", line=dict(color='#d62728', width=2, dash='dot')))
    
    lossPlot.update_layout(
        template="plotly_white",
        title="Model Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    lossPlot.show()

    # Plot training accuraccy
    accPlot = go.Figure()
    accPlot.add_trace(go.Scatter(x=df["Epoch"], y=df["Train Accuracy"], name="Train Accuracy", line=dict(color='#2ca02c', width=2)))
    accPlot.add_trace(go.Scatter(x=df["Epoch"], y=df["Test Accuracy"], name="Test Accuracy", line=dict(color='#9467bd', width=2, dash='dot')))
    
    accPlot.update_layout(
        template="plotly_white",
        title="Model Accuracy",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        hovermode="x unified",
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
    )
    accPlot.show()

    plots_path = os.path.join(os.path.dirname(file), "Plots")
    loss_path = os.path.join(plots_path, f"{filename}_loss.png")
    lossPlot.write_image(loss_path)
    acc_path = os.path.join(plots_path, f"{filename}_accuracy.png")
    accPlot.write_image(acc_path)

generatePlots(path, numEpochs=10000)