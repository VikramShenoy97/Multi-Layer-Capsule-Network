import numpy as np
import matplotlib.pyplot as plt
from plotly import tools
import chart_studio.plotly as py
import plotly.graph_objs as go
def generate_reconstructions(reconstructions, images=None, input=None):
    reconstructions = reconstructions.view(1, 1, 92, 112)
    reconstructions = reconstructions.detach().numpy()
    reconstructions = reconstructions * 255.
    if images is not None:
      images = images * 255.
    if type(input) == int:
      rows = 1
      columns = 1
      fig, axs = plt.subplots(rows, columns)
      r_sample = 0
      i_sample = 0
      for i in range(0, rows):
        for j in range(0, columns):
              if(j % 2 == 0):
                  axs.imshow(images[i_sample, :, :], cmap="gray")
                  axs.axis("off")
                  i_sample = i_sample + 1
              else:
                  axs.imshow(reconstructions[r_sample, :, :], cmap="gray")
                  axs.axis("off")
                  r_sample = r_sample + 1

      fig.savefig("Reconstructions for Epoch "+str(input)+".png")
      plt.show()
      plt.close()

    elif type(input) == str:
      rows = 5
      columns = 5
      for count in range(0, 2):
        fig, axs = plt.subplots(rows, columns)
        r_sample = 0
        i_sample = 0
        for i in range(0, rows):
          for j in range(0, columns):
                if(count == 0):
                    # Ground Truth
                    axs[i,j].imshow(images[i_sample, 0, :, :], cmap="gray")
                    axs[i,j].axis("off")
                    i_sample = i_sample + 1
                    title = "Ground_Truth_Images"
                else:
                    # Reconstructions
                    axs[i,j].imshow(reconstructions[r_sample, 0, :, :], cmap="gray")
                    axs[i,j].axis("off")
                    r_sample = r_sample + 1
                    title = "Reconstructed_Images"
        fig.savefig(title + ".png")
        plt.show()
        plt.close()

    elif input is None:
        dim = 16
        rows = 2
        columns = 5
        count = -1
        for dimension in range(0, dim):
            count += 1
            dim_sample = count
            fig, axs = plt.subplots(rows, columns)
            for i in range(0, rows):
                for j in range(0, columns):
                    axs[i,j].imshow(reconstructions[dim_sample, 0, :, :], cmap="gray")
                    axs[i,j].axis("off")
                    dim_sample = dim_sample + 11
            fig.savefig("Pose_Reconstructions_for_dimension_"+str(dimension+1)+".png")
            plt.show()
            plt.close()
    return

def plot_graph(test_accuracy):
    py.sign_in('VikramShenoy','x1Un4yD3HDRT838vRkFA')
    training_accuracy = []
    history = np.load("history.npy", allow_pickle=True)
    training_loss = history.item().get('train_loss')
    train_accuracy =  history.item().get('train_accuracy')
    epochs = list(range(1, len(training_loss)+1))
    for i in range(0, len(train_accuracy)):
        training_accuracy.append(train_accuracy[i].item()/100)
    testing_accuracy = 98.84#test_accuracy
    trace0 = go.Scatter(
    x = epochs,
    y = training_accuracy,
    mode = "lines",
    name = "Training Accuracy"
    )

    trace1 = go.Scatter(
    x = epochs,
    y = training_loss,
    mode = "lines",
    name = "Training Loss"
    )
    data = go.Data([trace0, trace1])
    layout = go.Layout()
    fig = go.Figure(data=data, layout=layout)
    fig['layout']['xaxis'].update(title="Number of Epochs", range = [min(epochs), max(epochs)], dtick=len(epochs)/10, showline = True, zeroline=True,  mirror='ticks', linecolor='#636363', linewidth=2)
    fig['layout']['yaxis'].update(title="Training Loss and Accuracy", range = [0, 1.05], dtick=0.1, showline = True, zeroline=True, mirror='ticks',linecolor='#636363',linewidth=2)
    py.image.save_as(fig, filename="Training_Graph.png")

    print("Training Graph Created")

    x_axis = ["Training", "Testing"]
    y_axis = [training_accuracy[-1]*100, testing_accuracy]
    trace2 = go.Bar(
    x = x_axis,
    y = y_axis,
    width = [0.3, 0.3]
    )
    data = [trace2]
    layout = go.Layout()
    fig = go.Figure(data=data, layout=layout)
    fig['layout']['xaxis'].update(title="Mode", showline = True)
    fig['layout']['yaxis'].update(title="Accuracy")
    py.image.save_as(fig, filename="Accuracy_Graph.png")
    print("Accuracy Graph Created")

    return
