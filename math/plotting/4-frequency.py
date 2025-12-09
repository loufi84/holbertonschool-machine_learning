#!/usr/bin/env python3
"""
This module contains a function that draw a histogram of student scores for
a project.
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    The function that draw the histogram.
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.hist(student_grades, bins=10, edgecolor="black")
    plt.show()
