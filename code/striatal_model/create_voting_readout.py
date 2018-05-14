import numpy as np
import pylab as plt
import json


def angular_voting_readout():
    num_neurons = 16

    angle = np.linspace(0, np.pi * 2, num_neurons)

    translation = list(np.sin(angle) / (num_neurons / 10.))
    rotation = list(np.cos(angle) / (num_neurons / 10.))

    weights = [translation, rotation]

    print len(translation)

    print
    print
    print len(rotation)

    print np.shape(weights)
    json.dump(weights, open("angular_voting_readout.dat", "w+"))

    plt.plot(rotation)
    plt.plot(translation)
    plt.show()


def population_voting_readout():

    translation = []
    rotation = []

    for j in range(80):

        if j < 40:
            # translation.append(1/2000.)
            translation.append(1 / 500.)
            rotation.append(1 / 260.)
        elif j < 80:
            # rotation.append(1/500.)
            rotation.append(-1 / 260.)
            translation.append(1 / 500.)

    weights = [translation, rotation]

    print translation

    print
    print
    print rotation
    print np.shape(weights)
    json.dump(weights, open("population_voting_readout.dat", "w+"))

    # plt.plot(rotation)
    # plt.plot(translation)
    # plt.show()


def channel_readout():

    N_D1 = N_D2 = 40
    S_D1 = 2.0
    S_D2 = 0.4

    go_left = []
    go_right = []

    for j in range(160):

        if j < 40:  # d1 left
            #go_left.append(1 / 50.)
            go_left.append(S_D1 / N_D1)
            go_right.append(0.)
        elif j < 80:  # d2 left
            go_left.append(-S_D2 / N_D2)
            go_right.append(0.)
        elif j < 120:  # d1 right
            go_left.append(0.)
            #go_right.append(1 / 50.)
            go_right.append(S_D1 / N_D1)
        elif j < 160:  # d2 right
            go_left.append(0.)
            go_right.append(-S_D2 / N_D2)

    weights = [go_left, go_right]
    json.dump(weights, open("channel_readout.dat", "w+"))


def channel_to_twist():
    translation = []
    rotation = []

    # speed = go_left + go_right
    translation.append(0.5)  # go left
    translation.append(0.5)  # go right

    # rotation = go_left - go_right
    rotation.append(1.)  # go_left
    rotation.append(-1.)  # go_right

    weights = [translation, rotation]
    json.dump(weights, open("channel_to_twist.dat", "w+"))


# angular_voting_readout()
# population_voting_readout()
channel_readout()
channel_to_twist()
