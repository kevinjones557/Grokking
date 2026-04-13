import os
import re
import math
import pandas as pd
import plotly.graph_objects as plotly

path = "/Users/pjhenwood/VS Projects/CS520/Final Project/Grokking/log_1775055592"

def generatePlots(file: str, numEpochs=math.inf):
    data = []
    regex = r"Epoch (\d+): Loss -- Train: ([\d.]+) Test: ([\d.]+)    Accuracy -- Train: ([\d.]+) Test: ([\d.]+)"
    
    for line in open(file, 'r'):
        match = re.search(regex, line)
        if match and int(match.group(1)) <= numEpochs:
            data.append({"Epoch": int(match.group(1)), "Train Loss": float(match.group(2)),
                        "Test Loss": float(match.group(3)), "Train Accuracy": float(match.group(4)),
                        "Test Accuracy": float(match.group(5))})
    
    df = pd.DataFrame(data)
    filename = os.path.basename(file)
    
    # Plot loss
    lossPlot = plotly.Figure()
    lossPlot.add_trace(plotly.Scatter(x=df["Epoch"], y=df["Train Loss"], name="Train Loss", line=dict(color="#6E5766", width=3)))
    lossPlot.add_trace(plotly.Scatter(x=df["Epoch"], y=df["Test Loss"], name="Test Loss", line=dict(color="#FF0000", width=3, dash='dot')))
    
    lossPlot.update_layout(
        template="plotly_white",
        height = 675,
        width = 900,
        title="Model Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        font=dict(family="sans-serif", size=18),
        legend=dict(yanchor="top", y=1.1, xanchor="right", x=0.99)
    )
    #lossPlot.show()

    # Plot training accuraccy
    accPlot = plotly.Figure()
    accPlot.add_trace(plotly.Scatter(x=df["Epoch"], y=df["Train Accuracy"], name="Train Accuracy", line=dict(color="#1DC750", width=3)))
    accPlot.add_trace(plotly.Scatter(x=df["Epoch"], y=df["Test Accuracy"], name="Test Accuracy", line=dict(color="#164CC2", width=3, dash='dot')))
    
    accPlot.update_layout(
        template="plotly_white",
        height = 675,
        width = 900,
        title="Model Accuracy",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        font=dict(family="sans-serif", size=18),
        legend=dict(yanchor="bottom", y=1, xanchor="right", x=1)
    )
    #accPlot.show()

    plots_path = os.path.join(os.path.dirname(file), "Plots")
    loss_path = os.path.join(plots_path, f"{filename}_loss.pdf")
    lossPlot.write_image(loss_path)
    acc_path = os.path.join(plots_path, f"{filename}_accuracy.pdf")
    accPlot.write_image(acc_path)

generatePlots(path)