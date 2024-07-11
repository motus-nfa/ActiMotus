import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from reportlab.pdfgen import canvas
from reportlab.platypus import Table
from IPython.display import display, HTML


def resolve_sanity_check_pre(ts_list, data_list):
    # Check if data_list is a list of numpy arrays
    if not isinstance(data_list, list) or not all(
        isinstance(i, np.ndarray) for i in data_list
    ):
        data_list = [data_list]
        ts_list = [ts_list]
    return ts_list, data_list


def sanity_check_pre(ts_list, data_list):
    # Check if both ts_list and data_list are lists
    if not (isinstance(ts_list, list) and isinstance(data_list, list)):
        # print(
        #     "ts_list and data_list are not list objects. Please make sure they are lists in the future. For this session we will convert them to lists for you."
        # )
        return resolve_sanity_check_pre(ts_list, data_list)

    return ts_list, data_list


def time_sanity(array):
    """
    Checks if the input array is flattened. If not, it flattens the array.

    Parameters:
    array (numpy.ndarray): The input array to check and possibly flatten.

    Returns:
    numpy.ndarray: The flattened array.
    """
    if len(array.shape) == 1:
        # print("The time array is flattened, proceeding...")
        pass
    else:
        print(
            # "Please keep in mind that the time array, aka, ts_list, should be passed as a flattened array.\n \n For this session we will flatten it for you."
        )
        array = array.flatten()
    return array


# ======================================================================= generate_csv  =====================================================================


# Generate csv report - tailored for alg_chunk_acti4pre_v1_1_3.py
def acti4pre_csv(ts_chunked, out_cat, out_val, i, sensor_id):
    # Create a safe sensor_id by replacing special characters with underscores
    safe_sensor_id = re.sub(r"[^\w]", "_", sensor_id)

    # Convert ts_chunked
    datetime_objects = [
        datetime.utcfromtimestamp(t / 1000).strftime("%Y-%m-%d %H:%M:%S.%f")
        for t in ts_chunked
    ]

    # Flatten out_cat
    flatten_out_cat = out_cat.flatten()

    # Convert out_val to dataframe
    out_val_df = pd.DataFrame(
        out_val,
        columns=[
            "Stdx",
            "Stdy",
            "Stdz",
            "Meanx",
            "Meany",
            "Meanz",
            "hlratio",
            "Iws",
            "Irun",
            "NonWear",
        ],
    )

    # Concatenate columns
    acti4pre = pd.concat(
        [
            pd.Series(
                ts_chunked, name="ts_chunked"
            ),  # This is a numpy array of timestamps in milliseconds
            pd.Series(
                datetime_objects, name="ts_chunked_iso"
            ),  # This is a list of timestamps in ISO format
            pd.Series(flatten_out_cat, name="out_cat"),
            out_val_df,
        ],
        axis=1,
    )

    # check if a folder under this name Pre_Acti4_V_1_1_3 exists, if it does, then append the csv file to the folder, if it doesn't then create the folder and then append the csv file to the folder
    if os.path.exists(f"{safe_sensor_id}\\Pre_Acti4_V_1_2_0"):
        # Append to CSV
        acti4pre.to_csv(
            f"{safe_sensor_id}\\Pre_Acti4_V_1_2_0\\pre_chunk_{i+1}.csv", index=False
        )
    else:
        # Create folder
        os.mkdir(f"{safe_sensor_id}\\Pre_Acti4_V_1_2_0")
        # Append to CSV
        acti4pre.to_csv(
            f"{safe_sensor_id}\\Pre_Acti4_V_1_2_0\\pre_chunk_{i+1}.csv", index=False
        )

    return acti4pre


# ===========================================================================================================================================================


# Generate csv report - tailored for alg_chunk_acti4pre_v1_1_3.py
def motuspre_merge_outputs(ts_chunked, out_cat, out_val, i):
    # Convert ts_chunked
    datetime_objects = [
        datetime.utcfromtimestamp(t / 1000).strftime("%Y-%m-%d %H:%M:%S.%f")
        for t in ts_chunked
    ]

    # Flatten out_cat
    flatten_out_cat = out_cat.flatten()

    # Convert out_val to dataframe
    out_val_df = pd.DataFrame(
        out_val,
        columns=[
            "Stdx",
            "Stdy",
            "Stdz",
            "Meanx",
            "Meany",
            "Meanz",
            "hlratio",
            "Iws",
            "Irun",
            "NonWear",
            "xsum",
            "zsum",
            "xSqsum",
            "zSqsum",
            "xzsum",
            "SF12",
        ],
    )

    # Concatenate columns
    acti4pre = pd.concat(
        [
            pd.Series(
                ts_chunked, name="ts_chunked"
            ),  # This is a numpy array of timestamps in milliseconds
            pd.Series(
                datetime_objects, name="ts_chunked_iso"
            ),  # This is a list of timestamps in ISO format
            pd.Series(flatten_out_cat, name="out_cat"),
            out_val_df,
        ],
        axis=1,
    )

    return acti4pre


# ===========================================================================================================================================================

# Generate csv report - tailored for alg_activity_acti4_v1_1_3.py


def activity_motus_csv(out_ts, out_cat, out_val, day, selected_id, res_folder, ver):
    # Create a safe sensor_id by replacing special characters with underscores
    # safe_sensor_id = re.sub(r'[^\w]', '_', sensor_id)

    # Convert ts_chunked
    datetime_objects = [
        datetime.utcfromtimestamp(int(t / 1000)).strftime("%Y-%m-%d %H:%M:%S.%f")
        for t in out_ts
    ]

    cols = [
        "out_ts",
        "out_cat",
        "Steps",
    ]
    if ver == "3sensors":
        cols += [
            "TrunkInc",
            "TrunkFB",
            "TrunkLat",
            "ArmInc",
        ]

    # Create a DataFrame using the arrays
    data = np.concatenate([out_ts, out_cat, out_val], axis=1)
    activity_motus = pd.DataFrame(data, columns=cols)
    activity_motus["out_ts_iso"] = datetime_objects
    # Create the directory if it doesn't exist
    directory = os.path.join(res_folder, f"{selected_id}\\poststep")
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Append to CSV
    activity_motus.to_csv(
        f"{directory}\\activity_chunk_{day:%Y_%m_%d}.csv", index=False
    )

    return activity_motus


# ===========================================================================================================================================================


def plot_activity_classification(activity_acti4, plot_title):
    # Define color and activity mapping
    color_dict = {
        1: "cyan",
        2: "green",
        3: "red",
        4: "purple",
        5: "orange",
        6: "yellow",
        7: "pink",
        8: "brown",
        9: "black",
    }
    activity_dict = {
        1: "lie",
        2: "sit",
        3: "stand",
        4: "move",
        5: "walk",
        6: "run",
        7: "stair",
        8: "cycle",
        9: "row",
    }

    # Create a new column for activity names
    activity_acti4["activity_name"] = activity_acti4["out_cat"].map(activity_dict)

    # Create a figure
    fig = go.Figure()

    # Create a scatter plot for each activity
    for activity in activity_dict.values():
        activity_data = activity_acti4[activity_acti4["activity_name"] == activity]
        if not activity_data.empty:  # Check if activity_data is not empty
            scatter = go.Scatter(
                x=activity_data["out_ts"],
                y=activity_data["out_cat"],
                mode="markers",
                name=activity,
                marker=dict(color=color_dict[activity_data["out_cat"].iloc[0]]),
            )  # map category to color
            fig.add_trace(scatter)

    # Create a line plot
    line = go.Scatter(
        x=activity_acti4["out_ts"],
        y=activity_acti4["out_cat"],
        mode="lines",
        name="Activity Line",
        line=dict(color="blue"),
    )  # set line color

    fig.add_trace(line)

    # Set the title
    fig.update_layout(
        title=f"{plot_title}", xaxis_title="Timestamp", yaxis_title="Activity Category"
    )

    fig.show()

    # Set the maximum width to None to display all columns
    pd.set_option("display.max_columns", None)

    # Generate statistical summary for each activity
    activity_summary = activity_acti4.groupby("activity_name").describe()

    return activity_summary


# ===========================================================================================================================================================


def display_dataframe_info(df):
    info = {
        "Dataframe Type": type(df),
        "Dataframe Shape": df.shape,
        "Dataframe Columns": df.columns.tolist(),
        "Dataframe Info": df.info(),
    }
    return info


# ===========================================================================================================================================================


def extract_sensor_id(file_path):
    """
    Extracts the sensor id from a file path.

    Args:
        file_path (str): The path of the file.

    Returns:
        str or None: The extracted sensor id if found, None otherwise.
    """
    # Get the base name of the file
    base_name = os.path.basename(file_path)

    # Use a regular expression to extract the sensor id
    match = re.search(r"export_(.*?)_acc", base_name)

    # If a match was found, return the sensor id
    if match:
        return match.group(1)
    else:
        match = re.search(r"export_raw_(\d+)", base_name)
        if match:
            return match.group(1)
        else:
            print(
                "No sensor id was found in the file path.\n\n Please make sure the file path is correct and try again."
            )
            return None


# ===========================================================================================================================================================


def extract_id_placements(id, src_folder):
    files = [i for i in os.listdir(src_folder) if id in i and ".bin" in i]
    try:
        file_plc = [i.split("-")[1].replace(".bin", "") for i in files]
    except:
        file_plc = ["thigh"]
    files = {
        {"lår": "thigh", "ryg": "trunk", "læg": "calf", "arm": "arm"}.get(
            file_plc[i].lower(), file_plc[i].lower()
        ): file
        for i, file in enumerate(files)
    }
    return files


# ===========================================================================================================================================================


# function for generating range of timestamps from first chunk to last (across all placements)
def generate_timestamp_range(start_timestamp, end_timestamp):
    current_timestamp = start_timestamp.replace(microsecond=0)
    while current_timestamp <= end_timestamp:
        yield current_timestamp
        current_timestamp += timedelta(hours=12)


# function to add index of list of chunk where sensor has first data from
def get_chunk_start_idx(chunk_dict, selected_placements):
    # lists to store first timestamps from first and last chunk
    start_first = []
    start_last = []
    for plc in selected_placements:
        # Get timestamps from first and last chunk (add 10 minutes do to overlaps)
        chunk_start = chunk_dict[plc]["list_of_df"][0].iloc[0, 0] + timedelta(
            minutes=10
        )
        chunk_end = chunk_dict[plc]["list_of_df"][-1].iloc[0, 0] + timedelta(minutes=10)
        # Floor to timestamp to 12:00 or 00:00
        chunk_start_clean = chunk_start - timedelta(
            hours=chunk_start.hour % 12,
            minutes=chunk_start.minute,
            seconds=chunk_start.second,
        )
        chunk_end_clean = chunk_end - timedelta(
            hours=chunk_end.hour % 12,
            minutes=chunk_end.minute,
            seconds=chunk_end.second,
        )
        # Append to lists
        start_first.append(chunk_start_clean)
        start_last.append(chunk_end_clean)

    # Find first chunk and last chunk across all placements
    start_timestamp = min(start_first)
    end_timestamp = max(start_last)

    # Generate the range of timestamps from first chunk to last chunk across all placements
    timestamp_range = list(generate_timestamp_range(start_timestamp, end_timestamp))

    start_idx = {}

    # For each placement find out where in above range first chunk appears
    for plc in selected_placements:
        for i, ts in enumerate(timestamp_range):
            chunk_start = chunk_dict[plc]["list_of_df"][0].iloc[0, 0] + timedelta(
                minutes=10
            )
            if chunk_start < ts + timedelta(hours=12):
                start_idx[plc] = i
                break

    # Add Nones before in list of chunks
    for plc in selected_placements:
        miss_chunk_end = len(timestamp_range) - (
            start_idx[plc] + len(chunk_dict[plc]["list_of_df"])
        )
        chunk_dict[plc]["list_of_df"] = (
            [
                None for i in range(start_idx[plc])
            ]  # Add Nones for chunks before first chunk
            + chunk_dict[plc]["list_of_df"]  # Add chunks
            + [
                None for i in range(miss_chunk_end)
            ]  # Add Nones for chunks after last chunk
        )
        chunk_dict[plc]["extended_chunks_original_format"] = (
            [
                None for i in range(start_idx[plc])
            ]  # Add Nones for chunks before first chunk
            + chunk_dict[plc]["extended_chunks_original_format"]  # Add chunks
            + [
                None for i in range(miss_chunk_end)
            ]  # Add Nones for chunks after last chunk
        )

    return chunk_dict, timestamp_range


# ===========================================================================================================================================================


def chunk_data_extended(read_bin_data, desired_chunking, overlapping_threshold):
    """
    This function processes the input data, calculates the sampling frequency,
    chunks the data into specified hours, and prints the first chunk. It also includes data from 10 minutes before and after each chunk.

    Parameters:
    read_bin_data (numpy array): The input data to be processed. It should have four columns: 'timestamp', 'x', 'y', 'z'.
    desired_chunking (int): The desired chunking in hours, like 6, 12, 24, etc.
    overlapping_threshold (int): The desired overlapping threshold in minutes, like 10, 20, 30, etc.

    Returns:
    list_of_df (list): A list of DataFrames, each representing a chunk of the original data.
    sampling_frequency (float): The calculated sampling frequency.
    df (pandas.DataFrame): The original DataFrame.
    """

    # Create DataFrame from the input data
    df = pd.DataFrame(read_bin_data, columns=["timestamp", "x", "y", "z"])

    # Convert the timestamp column to datetime and create a new column timestamp_unix
    df["timestamp_unix"] = df["timestamp"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Reorder the columns
    df = df[["timestamp", "x", "y", "z", "timestamp_unix"]]

    # Group by the desired chunking period without setting timestamp as index
    list_of_df = [
        group
        for _, group in df.groupby(
            pd.Grouper(key="timestamp", freq=f"{desired_chunking}H")
        )
    ]

    # Calculate the difference between consecutive timestamps and get the frequency
    sampling_frequency = 1 / df["timestamp"].diff().dt.total_seconds().mean()

    # print(f"The sampling frequency is {sampling_frequency} Hz.\n")
    # print(f"There is a total of {len(list_of_df)} chunks available , meaning that the dataset is sliced into {len(list_of_df)} parts wheer each part contains {desired_chunking} hours of data.\n")

    # Display the available chunks based on the desired chunking and their date and time
    # print("The available chunks are: \n")
    colors = [
        "darkred",
        "darkcyan",
        "darkmagenta",
        "darkkhaki",
        "darkgray",
        "darkolivegreen",
        "darkgoldenrod",
        "darkslateblue",
        "darkslategray",
    ]

    # Create a dictionary that maps each unique date to a color
    unique_dates = df["timestamp"].dt.date.unique()
    date_color_map = {
        date: colors[i % len(colors)] for i, date in enumerate(unique_dates)
    }

    # for i in range(len(list_of_df)):
    #     date = list_of_df[i]['timestamp'].iloc[0].date()
    #     color = date_color_map[date]  # Get the color for the current date
    #     display(HTML(f"<text style=color:{color}>Chunk {i+1}: {list_of_df[i]['timestamp'].iloc[0]} to {list_of_df[i]['timestamp'].iloc[-1]}</text>"))

    # Create a new list to store the extended chunks
    extended_chunks = []

    # Iterate over the chunks
    for i in range(len(list_of_df)):
        # Get the start and end timestamps of the current chunk
        start_time = list_of_df[i]["timestamp"].min()
        end_time = list_of_df[i]["timestamp"].max()

        # Extend the start and end times by xx minutes
        extended_start_time = start_time - pd.Timedelta(minutes=overlapping_threshold)
        extended_end_time = end_time + pd.Timedelta(minutes=overlapping_threshold)

        # Filter the original DataFrame to include data from the extended time range
        extended_chunk = df[
            (df["timestamp"] >= extended_start_time)
            & (df["timestamp"] <= extended_end_time)
        ]

        # Append the extended chunk to the list
        extended_chunks.append(extended_chunk)

    # print("\n\nFor instance, the first chunk is: \n", extended_chunks[0])

    # Create a new list to store the extended chunks in the original format
    extended_chunks_original_format = []

    # Iterate over the extended chunks
    for chunk in extended_chunks:
        # Select only the 'timestamp_unix', 'x', 'y', and 'z' columns, convert the 'timestamp_unix' column to int64, convert the DataFrame to a numpy array, and append it to the list
        extended_chunks_original_format.append(
            chunk[["timestamp_unix", "x", "y", "z"]]
            .astype({"timestamp_unix": "int64"})
            .values
        )

    # print("\n\nFor instance, the first chunk in the original format is: \n", extended_chunks_original_format[0])

    return extended_chunks, sampling_frequency, df, extended_chunks_original_format


# ===========================================================================================================================================================


def add_table_to_pdf(
    c,
    df,
    title,
    current_y,
    page_width,
    page_height,
    margin,
    title_table_distance,
    each_section_distace,
):
    """
    This function adds a DataFrame to the PDF.

    Parameters:
    c (canvas.Canvas): The canvas object to draw on.
    df (pandas.DataFrame): The DataFrame to be added to the PDF.
    title (str): The title of the table.
    current_y (float): The current y-coordinate of the canvas.
    page_width (float): The width of the page.
    margin (float): The margin of the page.
    title_table_distance (float): The distance between the title and the table.

    Returns:
    current_y (float): The updated y-coordinate of the canvas.
    """

    # Set the title for the table
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(page_width / 2, current_y - 70, title)

    # Subtract the height of the title and some space for the gap
    current_y -= 70 + title_table_distance

    # Add the df to the PDF
    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data)

    # Set the table style
    table.setStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), "gray"),
            ("TEXTCOLOR", (0, 0), (-1, 0), "white"),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, "black"),
            ("BOX", (0, 0), (-1, -1), 0.25, "black"),
        ]
    )

    # Set the table size
    w, h = table.wrapOn(c, page_width - 2 * margin, page_height - 2 * margin)

    # Calculate the x-coordinate for the table to be placed in the middle of the page
    x = (page_width - w) / 2

    # Draw the table on the canvas
    table.drawOn(c, x, current_y - h)

    # Subtract the height of the table and some space for the gap
    current_y -= h + title_table_distance - each_section_distace

    return current_y


# ===========================================================================================================================================================


def generate_pdf(list_of_df, sampling_frequency, df, name):
    """
    This function generates a PDF report for the input data.

    Parameters:
    list_of_df (list): A list of DataFrames, each representing a chunk of the original data.

    Returns:
    a PDF report Saved in the current directory.
    """

    title_table_distance = 10  # The distance between the title and the table
    each_section_distace = 40  # The distance between each section

    # ================================================ PDF REPORT Page setting ================================================

    # create a folder named results if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")

    # Create a new canvas object
    c = canvas.Canvas(f"results\\{extract_sensor_id(name)}\\report.pdf")

    # set the page size , A4
    page_width, page_height = (595.27, 841.89)
    c.setPageSize((page_width, page_height))

    # Define margins (20mm converted to points)
    margin = 20 * 2.83465  # 1mm is approximately 2.83465 points

    # add a title and set the title left aligned
    c.setFont("Helvetica-Bold", 14)
    title_y = page_height - margin
    c.drawString(margin, title_y, "Data Chunking Report for Acti4 Sandbox Environment")

    # move the lego on the y-axis 10mm up
    title_y -= -10 * 2.83465

    # add a logo, right aligned
    logo_width = 50
    logo_height = 50
    logo_x = page_width - margin - logo_width
    c.drawImage(
        ".\\BackendFiles\\Extras\\sens logo black.png",
        logo_x,
        title_y - logo_height,
        width=logo_width,
        height=logo_height,
        mask="auto",
        preserveAspectRatio=True,
    )

    # add a horizontal line - line thickness 1mm
    c.setLineWidth(1)
    c.line(
        margin,
        page_height - margin - 10,
        page_width - margin,
        page_height - margin - 10,
    )

    # ==================================================== Sensor Information ================================================

    # Start at the top of the page
    current_y = page_height - margin

    # Create a DataFrame with the sensor id, sampling frequency, start and end information
    info_df = pd.DataFrame(
        {
            "Sensor ID": [extract_sensor_id(name)],
            "Sampling Frequency": [f"{sampling_frequency:.3f} Hz"],
            "Start Date and Time": [df["timestamp"].iloc[0]],
            "End Date and Time": [df["timestamp"].iloc[-1]],
        }
    )

    # Use the helper function to add the table to the PDF
    current_y = add_table_to_pdf(
        c,
        info_df,
        "Sensor Information",
        current_y,
        page_width,
        page_height,
        margin,
        title_table_distance,
        each_section_distace,
    )

    #  ================================================ Data Chunks Information ================================================

    # Create a DataFrame with the chunk number and the start and end times of each chunk
    chunks_df = pd.DataFrame(
        {
            "Chunk Number": [i + 1 for i in range(len(list_of_df))],
            "Start Time": [chunk["timestamp"].iloc[0] for chunk in list_of_df],
            "End Time": [chunk["timestamp"].iloc[-1] for chunk in list_of_df],
        }
    )

    # Keep only the first 2 and last 2 rows, and add a row with '...' in between
    chunks_df = pd.concat(
        [
            chunks_df.head(2),
            pd.DataFrame(
                {"Chunk Number": ["..."], "Start Time": ["..."], "End Time": ["..."]}
            ),
            chunks_df.tail(2),
        ]
    )

    # Use the helper function to add the table to the PDF
    current_y = add_table_to_pdf(
        c,
        chunks_df,
        "Data Chunks Information",
        current_y,
        page_width,
        page_height,
        margin,
        title_table_distance,
        each_section_distace,
    )

    # ========================================== Data Samples from the First Chunk ===========================================

    # Assuming list_of_df[0] is your DataFrame - selecting the first ten and last ten rows to fit in the PDF
    df_top = list_of_df[0][0:3]
    df_bottom = list_of_df[0][-3:]

    # Create a DataFrame for the indicator and concatenate it three times
    for _ in range(1):
        df_indicator = pd.DataFrame(
            {col: ["..."] for col in df_top.columns}, index=[df_top.index[-1] + 1]
        )
        df_top = pd.concat([df_top, df_indicator])

    # Concatenate the top and bottom DataFrames
    df = pd.concat([df_top, df_bottom])

    # Use the helper function to add the table to the PDF
    current_y = add_table_to_pdf(
        c,
        df,
        "Data Samples from the First Chunk",
        current_y,
        page_width,
        page_height,
        margin,
        title_table_distance,
        each_section_distace,
    )

    # ================================================ Data Visualization ================================================

    # Create a new plot with a title and axis labels
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=list_of_df[0], x="timestamp", y="x", label="x", color="salmon")
    sns.lineplot(
        data=list_of_df[0], x="timestamp", y="y", label="y", color="mediumseagreen"
    )
    sns.lineplot(data=list_of_df[0], x="timestamp", y="z", label="z", color="steelblue")
    plt.xlabel("Time")
    plt.ylabel("Acceleration")
    plt.legend()

    # make a directory under sensor name if it doesn't exist
    if not os.path.exists(f"results\\{extract_sensor_id(name)}"):
        os.makedirs(f"results\\{extract_sensor_id(name)}")

    # Save the figure
    plt.savefig(
        f"results\\{extract_sensor_id(name)}\\raw_acc.png",
        bbox_inches="tight",
        pad_inches=0.25,
    )

    # Close the figure
    plt.close()

    # Add a title and set the title center for the data visualization
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(
        page_width / 2, current_y - 70, "Acceleration over Time - First Chunk"
    )

    # Subtract the height of the title and some space for the gap
    current_y -= 62 + title_table_distance

    # Add the figure to the PDF
    img_width = page_width - 2 * margin
    img_height = 220
    img_x = margin
    img_y = current_y - img_height
    c.drawImage(
        f"results\\{extract_sensor_id(name)}\\raw_acc.png",
        img_x,
        img_y,
        width=img_width,
        height=img_height,
        mask="auto",
    )

    # Subtractf the hei\\{extract_sensor_id(name)}ght of the figure and some space for the gap
    current_y -= img_height + title_table_distance

    # ================================================ add a page footer and logo ================================================

    # add a horizontal line - line thickness 1mm
    c.setLineWidth(0.5)
    c.line(margin, margin - 10, page_width - margin, margin - 10)

    # add the date and time of the report generation to the footer
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    c.setFont("Helvetica-Bold", 8)
    c.drawString(margin, margin - 22, f"Report generated on {dt_string}")

    # ================================================== END of PDF REPORT ==================================================

    # Save the PDF
    c.save()

    print(
        "The PDF report is generated successfully and saved in the current directory."
    )


# ===========================================================================================================================================================


def process_pre_step_data(out_pre_all_chunks):
    """
    Process the data by removing the first and last ten minutes of each chunk,
    converting timestamps to datetime, and converting the data to numpy arrays.

    Args:
        out_pre_all_chunks (list): List of chunks, where each chunk is a list of dictionaries.

    Returns:
        list: List of numpy arrays, where each array represents a modified chunk of data. where 10 minute overlapps are removed from the beginning and end of each chunk.
        The returned array is what step 2 of the algorithm expects as input.
    """
    rows_to_remove_ending = 10 * 60 - 2  # 10 minutes * 60 seconds
    rows_to_remove_beginning = 10 * 60  # 10 minutes * 60 seconds

    # Define column names
    column_names = [
        "ts",
        "cat",
        "stdx",
        "stdy",
        "stdz",
        "meanx",
        "meany",
        "meanz",
        "hlratio",
        "iws",
        "irun",
        "nonwear",
        "xsum",
        "zsum",
        "xSqsum",
        "zSqsum",
        "xzsum",
        "SF12",
    ]

    # Convert the list of lists to a list of dictionaries
    out_pre_all_chunks_dicts = [
        (
            list(map(lambda x: dict(zip(column_names, x)), chunk))
            if chunk is not None
            else None
        )
        for chunk in out_pre_all_chunks
    ]

    # Convert timestamp to datetime
    for chunk in out_pre_all_chunks_dicts:
        if chunk is not None:
            for row in chunk:
                row["datetime"] = datetime.utcfromtimestamp(row["ts"] / 1000).strftime(
                    "%Y-%m-%d %H:%M:%S.%f"
                )

    # Create an empty list to store the modified chunks
    out_pre_all_chunks_modified = []

    # Loop through the list of chunks
    for i in range(len(out_pre_all_chunks_dicts)):
        # If the chunk is the first chunk in the list
        if out_pre_all_chunks_dicts[i] is not None:
            if i == 0:
                # Remove the last ten minutes of data
                out_pre_all_chunks_modified.append(
                    out_pre_all_chunks_dicts[i][:-rows_to_remove_ending]
                )

            # If the chunk is the last chunk in the list
            elif i == len(out_pre_all_chunks_dicts) - 1:
                # Remove the first ten minutes of data
                out_pre_all_chunks_modified.append(
                    out_pre_all_chunks_dicts[i][rows_to_remove_beginning:]
                )

            # If the chunk is any other chunk in the list
            else:
                # Remove the first and last ten minutes of data
                out_pre_all_chunks_modified.append(
                    out_pre_all_chunks_dicts[i][
                        rows_to_remove_beginning:-rows_to_remove_ending
                    ]
                )
        else:
            out_pre_all_chunks_modified.append(None)

    # Create an empty list to store the numpy arrays
    out_pre_all_chunks_arrays = []

    for dictionary in out_pre_all_chunks_modified:
        # Convert list of dictionaries back to list of lists
        out_pre_all_chunks_lists = []
        if dictionary is not None:
            for row in dictionary:
                out_pre_all_chunks_lists.append([row[col] for col in column_names])
        else:
            out_pre_all_chunks_lists.append(None)

        # Convert list of lists to numpy array
        # out_pre_all_chunks_array = np.array(out_pre_all_chunks_lists, dtype=np.int64)
        out_pre_all_chunks_array = np.array(out_pre_all_chunks_lists)

        # store the numpy array in a list
        out_pre_all_chunks_arrays.append(out_pre_all_chunks_array)

    return out_pre_all_chunks_arrays


# ===========================================================================================================================================================
