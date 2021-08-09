import streamlit as st
import pandas as pd
import openpyxl
import numpy as np

np.set_printoptions(precision=3, suppress=True)
import matplotlib.pyplot as plt
import matplotlib
from math import log10, log2

st.title("Soil Classification Tool")
st.write("")
st.write("Please import a .csv file containing the mass retained on each sieve.")
st.write("This program was developed based on the U.S Sieve No. system.")
st.header("Instructions:")
st.header("Create an excel spreadsheet with two columns labeled "Particle Size" and "W-Retained"")
st.header("Write down the sizes of each sieve gradation and the corresponding weight retained")
st.header("The rest is taken care of ;))

from math import log2, log10


def log_interp(target, x1, x2, y1, y2):
    result = 10 ** (log10(y1) + (target - x1) * (log10(y2 / y1) / (x2 - x1)))
    return result


data = st.file_uploader("Test", type='xlsx')

if data:
    df = pd.read_excel(data)
    # st.dataframe(df)
    # st.table(df)
    total_mass = df['W-Retained'].sum()
    # given w-retained, the other parameters are easily computed
    # total soil mass = sum of all elements which user inputted
    # w-passing = Total Mass - (w-passing) @ row i
    # % retained = w-retained/total mass
    # % passing = 1 - % retained
    # plot particle size vs. % passing on log-log plot
    # follow decision tree to classify soil

    w_retained = df['W-Retained'].values

    w_cumulative = [[0]]
    for i in range(1, len(w_retained)):
        w_cumulative.append(w_retained[i] + w_cumulative[i - 1])

    w_cumulative = np.array(w_cumulative)

    num_elements = len(df)

    df['Cumulative Retained'] = w_cumulative

    percent_finer = [[1]]

    for i in range(1, len(w_cumulative)):
        percent_finer.append(1 - (w_cumulative[i] / total_mass.astype(float)))

    percent_finer = np.array(percent_finer)
    df['% Passing'] = percent_finer

    percent_retained = []
    for i in range(0, len(w_cumulative)):
        percent_retained.append(1 - percent_finer[i])

    percent_retained = np.array(percent_retained)
    df['% Retained'] = percent_retained

    st.table(df)
    st.write(f"Total Mass of Sample: {total_mass}")
    st.write("* Arbitrary Units")
    x = df['Particle Size'].values
    y = df['% Passing'].values
    fig = plt.figure()
    tick_y = np.arange(0, 100, 5)
    ax = fig.add_subplot(111)
    plt.xscale("log")
    ax.plot(x, y)
    plt.xlabel("Particle Size (*)")
    plt.ylabel("Percent Finer (%)")
    st.pyplot(fig)

    r_0075 = df.loc[df.isin([0.075]).any(axis=1)]  # Threshold for Coarse vs. Fine Soil
    r_0075 = r_0075['% Retained'].values

    r_4750 = df.loc[df.isin([4.75]).any(axis=1)]
    r_4750 = r_4750['% Retained'].values

    percent_fines = df.loc[df.isin([0.075]).any(axis=1)]
    percent_fines = percent_fines['% Passing'].values
    particle_size = np.array(df['Particle Size'])



    for i in range(1, len(percent_finer)):
        if percent_finer[i] < .30 and percent_finer[i - 1] > .30:
            x1_D30 = percent_finer[i]
            x2_D30 = percent_finer[i - 1]
            y1_D30 = particle_size[i]
            y2_D30 = particle_size[i - 1]
            D30 = log_interp(0.30, x1_D30, x2_D30, y1_D30, y2_D30)
            print("X1: ", x1_D30)
            print("X2: ", x2_D30)
            print("Y1: ", y1_D30)
            print("Y2: ", y2_D30)
            print("D30: ", D30)
    for i in range(1, len(percent_finer)):
        if percent_finer[i] < .10 and percent_finer[i - 1] > .10:
            x1_D10 = percent_finer[i]
            x2_D10 = percent_finer[i - 1]
            y1_D10 = particle_size[i]
            y2_D10 = particle_size[i - 1]
            D10 = log_interp(0.10, x1_D10, x2_D10, y1_D10, y2_D10)

    for i in range(1, len(percent_finer)):
        if percent_finer[i] < .60 and percent_finer[i - 1] > .60:
            x1_D60 = percent_finer[i]
            x2_D60 = percent_finer[i - 1]
            y1_D60 = particle_size[i]
            y2_D60 = particle_size[i - 1]
            D60 = log_interp(0.60, x1_D60, x2_D60, y1_D60, y2_D60)

    # Fine or Course
    if r_0075[0] > 0.5:
        property = "coarse"
        st.write(f"* The soil is {property}")
        st.write(f"* D10 = {D10}")
        st.write(f"* D30 = {D30}")
        st.write(f"* D60 = {D60}")
        Cc = (D30 ** 2) / (D60 * D10)
        Cu = D60 / D10
        st.write(f"* Cc = {Cc}")
        st.write(f"* Cu = {Cu}")

        if r_4750 / r_0075 > 0.5:
            property2 = "gravel"
            st.write(f"* The soil is a {property2}")
            if percent_fines[0] < 0.05:
                if Cu >= 4 and Cc >= 1 and Cc <= 3:
                    st.write("* Soil is GW: Well-Graded Gravel")
                elif Cu < 4 or Cc < 3:
                    st.write("* Soil is GP: Poorly-Graded Gravel")

            elif percent_fines[0] >= 0.05 and percent_fines[0] <= 0.12:
                st.write(f"* Soil contains relevant {percent_fines[0]}% fines")
                st.write("-> Requires dual classification. See plasticity chart")
                if Cu >= 4 and 1 <= Cc <= 3:
                    st.write("* First Label: Well-Graded Gravel (GW)")
                elif Cu < 4 and (Cc < 1 or Cc > 3):
                    st.write("* First Label: Poorly-Graded Gravel (GP)")


        else:
            property2 = "sand"
            st.write(f"* The soil is a {property2}")
            if percent_fines[0] < 0.05:
                if Cu >= 6 and 1 <= Cc <= 3:  # Cu >= 4 and Cc >= 1 and Cc <= 3:
                    st.write("* Soil is SW: Well-Graded Sand")
                elif Cu < 6 and (Cc < 1 or Cc > 3):
                    st.write("* Soil is SP: Poorly-Graded Sand")

            elif 0.05 <= percent_fines[0] <= 0.12:
                st.write(f"* Soil contains relevant {percent_fines[0]}% fines")
                st.write("Requires dual classification. See plasticity chart")
                if Cu >= 6 and 1 <= Cc <= 3:
                    st.write("* First Label: Well-Graded Sand (SW) ")
                elif Cu < 6 or (Cc < 1 or Cc > 3):
                    st.write("* First Label: Poorly-Graded Sand (SP)")
    else:
        st.write(f"* The soil is {property}")
        st.write("Please consult plasticity chart.")
