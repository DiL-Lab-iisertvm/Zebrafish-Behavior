import pandas as pd
import numpy as np

# Define the cutoffs for each region of interest (ROI)
cutoff_x_left = 0   # Adjust this value based on your ROI
cutoff_x_right = 579  # Adjust this value based on your ROI
cutoff_y = 304.5         # Y coordinate for cutoff

# Define likelihood threshold
likelihood_threshold = 0.9

# Load the CSV file with multi-level headers
df = pd.read_csv('Ntd3_AS13DLC_resnet50_NovelTank2Oct1shuffle1_100000.csv', header=[0, 1, 2])

# Check the structure of the DataFrame
#print(df.head())
#print(df.columns)

# Extract the model name and body parts
model_name = df.columns.levels[0][0]  # Get the model name
bodyparts = df.columns.levels[1][1:]   # Get the body parts, excluding the 'bodyparts' label

# Create lists for x, y, and likelihood columns
x_columns = [(model_name, bodypart, 'x') for bodypart in bodyparts]
y_columns = [(model_name, bodypart, 'y') for bodypart in bodyparts]
likelihood_columns = [(model_name, bodypart, 'likelihood') for bodypart in bodyparts]

# Print the columns to verify
print("X columns:", x_columns)
print("Y columns:", y_columns)
print("Likelihood columns:", likelihood_columns)
#print(body_part_data)
# Filter for valid rows based on likelihood
valid_rows = (df[likelihood_columns] > likelihood_threshold).all(axis=1)

# Get the valid x and y coordinates
x_coords = df[x_columns][valid_rows].values
y_coords = df[y_columns][valid_rows].values

# Initialize arrays to store bout times and transitions
time_spent = {bodypart: 0 for bodypart in bodyparts}  # Initialize time spent per bodypart
time_spent['ROI1'] = 0  # Add ROI1
time_spent['ROI2'] = 0  # Add ROI2
bouts = []  # To record bouts of time spent
transitions = 0

# Frame rate
frame_rate = 30.0
frame_time = 1.0 / frame_rate

# Variables to keep track of the current region
current_region = None
start_frame = None

for i in range(len(x_coords)):
    x = x_coords[i]
    y = y_coords[i]

    # Check if the points are valid
    if np.isnan(x).any() or np.isnan(y).any():
        continue  # Skip invalid points

    # Determine the current region based on x and y coordinates
    if (y < cutoff_y).all():  # ROI 1
        region = 'ROI1'
    elif (y >= cutoff_y).all():  # ROI 2
        region = 'ROI2'
    else:
        region = 'Outside'

    # Handle transitions
    if current_region is not None and current_region != region:
        transitions += 1  # Increment transition count
        if start_frame is not None:
            bout_duration = (i - start_frame) * frame_time
            bouts.append((current_region, bout_duration))  # Record the bout
            time_spent[current_region] += bout_duration  # Update time spent
        start_frame = i  # Reset start frame

    if region != 'Outside':
        if start_frame is None:  # If we're entering a region
            start_frame = i  # Mark start of the bout
    else:
        start_frame = None  # Reset if outside ROI

    current_region = region  # Update the current region

# Handle the last bout if still in a region
if current_region is not None and start_frame is not None:
    bout_duration = (len(x_coords) - start_frame) * frame_time
    bouts.append((current_region, bout_duration))
    time_spent[current_region] += bout_duration

# Output results
print(f'Time spent in each ROI: {time_spent}')
print(f'Number of transitions: {transitions}')
print('Bouts of time spent:', bouts)
