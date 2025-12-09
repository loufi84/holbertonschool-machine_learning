#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def bars():
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    people = ["Farrah", "Fred", "Felicia"]
    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]

    x = np.arange(len(people))

    plt.bar(x, apples, width=0.5, color="red", label="apples")
    plt.bar(x, bananas, width=0.5, bottom=apples, color="yellow", label="bananas")
    plt.bar(x, oranges, width=0.5, bottom=apples+bananas, color="#ff8000", label="oranges")
    plt.bar(x, peaches, width=0.5, bottom=apples+bananas+oranges, color="#ffe5b4", label="peaches")

    plt.ylabel("Quantity of Fruit")
    plt.title("Number of Fruit per Person")
    plt.xticks(x, people)
    plt.yticks(range(0, 81, 10))
    plt.ylim(0, 80)
    plt.legend()

    plt.show()
