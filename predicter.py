from bs4 import BeautifulSoup
import pandas as pd
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten

url = "https://www.euro-millions.com/pt/arquivo-de-resultados-200"

def getHtml(n):
    response = requests.get(url+str(n))

    if response.status_code == 200:
        html_content = response.text
    else:
        print("Failed to fetch the web page")
        html_content = None

    return html_content

def group_into_fives(input_list):
    return [input_list[i:i+5] for i in range(0, len(input_list), 5)]

def group_into_twos(input_list):
    return [input_list[i:i+2] for i in range(0, len(input_list), 2)]

def LSTM_model(df):
    # Convert the DataFrame to a NumPy array
    data = df.values.astype(np.float32)

    # Normalize the data to values between 0 and 1
    data /= 100

    # Create sequences for input and target data
    num_input_rows = 7
    num_target_rows = 7

    X, y = [], []
    for i in range(len(data) - num_input_rows - num_target_rows + 1):
        X.append(data[i:i+num_input_rows])
        y.append(data[i+num_input_rows:i+num_input_rows+num_target_rows])

    # Convert the sequences to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # Split the data into training and test sets
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Reshape y_train and y_test to match the model's output shape
    y_train = y_train.reshape(-1, num_target_rows * 7)
    y_test = y_test.reshape(-1, num_target_rows * 7)

    model = Sequential()
    model.add(LSTM(32, input_shape=(num_input_rows, 7)))
    model.add(Dense(num_target_rows * 7))  # Output layer with the correct number of neurons

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test, verbose=0)
    print("Test Loss:", loss)

    # Make predictions for the next sequence
    next_sequence = model.predict(np.expand_dims(X_test[-1], axis=0))

    # Denormalize the predictions to get the actual numbers
    next_sequence *= 100

    # Convert the predicted sequence to integers
    next_sequence = next_sequence.round().astype(int)

    # Reshape the predictions to match the original format
    next_sequence = next_sequence.reshape(num_target_rows, 7)

    # Print the predicted next sequence
    print("\nPredicted Next Sequence:\n", next_sequence)

if __name__ == "__main__":
    balls = []
    stars = []

    for i in range(4, 23):
        soup = BeautifulSoup(getHtml(4), 'html.parser')

        ball = soup.find_all(class_='resultBall ball small')
        star = soup.find_all(class_='resultBall lucky-star small')

        for element in ball:
            balls.append(element.text.strip())

        for element in star:
            stars.append(element.text.strip())

    balls = group_into_fives(balls)
    stars = group_into_twos(stars)

    #print(balls)
    #print(stars)

    # Make sure both lists have the same length
    assert len(balls) == len(stars), "Lists must have the same length"

    # Create an empty DataFrame with 7 columns
    df = pd.DataFrame(columns=['N1', 'N2', 'N3', 'N4', 'N5', 'S1', 'S2'])

    # Add each group as a new row in the DataFrame
    for i in range(len(balls)):
        row_data = balls[i] + stars[i]
        df.loc[i] = row_data

    LSTM_model(df)