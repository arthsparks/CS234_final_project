import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from generate_txt import generate_data

num_trials = 3
with_risk_sensitivity = False


def LinUCB(data, num_labels, shuffle, alpha):
    shuffled_data = data.sample(frac=1) if shuffle else data
    true_label, features = shuffled_data.iloc[:, 0], shuffled_data.iloc[:, 1:]
    features['augmentation'] = 1
    num_points, num_feature = features.shape
    A, b = {}, {}
    actions = []
    rewards = []
    p = np.zeros(num_labels)
    for label in range(num_labels):
        A[label] = np.identity(num_feature)
        b[label] = np.zeros(num_feature)
    for index, patient in features.iterrows():
        patient = patient.values
        patient = patient / np.linalg.norm(patient)
        for label in range(num_labels):
            A_inv = np.linalg.inv(A[label])
            theta = A_inv @ b[label]
            p[label] = theta @ patient + alpha * np.sqrt(patient @ A_inv @ patient)
            reward = compute_reward(label, true_label[index], with_risk_sensitivity)
            A[label] += np.outer(patient, patient)
            b[label] += reward * patient
            rewards.append(reward)
        prediction = np.random.choice(np.flatnonzero(p == p.max()))
        actions.append(prediction)
    print("LinUCB average return:", np.sum(actions == true_label.values) / num_points)
    print("# of bad actions:", calc_bad_actions(actions, true_label.values))
    return rewards


def compute_reward(prediction, true_label, with_risk=False):
    if with_risk:
        return compute_reward_with_risk(prediction, true_label)
    else:
        return 0 if prediction == true_label else -1


def compute_reward_with_risk(prediction, true_label):
    if abs(prediction - true_label) == 1:
        return -1
    if abs(prediction - true_label) == 2:
        return -20
    else:
        return 0


def calc_overall_rewards(actions, true_labels):
    assert(len(actions) == true_labels.size)
    overall_rewards = 0.
    for i in range(len(actions)):
        overall_rewards += compute_reward(actions[i], true_labels[i])
    return overall_rewards


def calc_bad_actions(actions, true_labels):
    assert(len(actions) == true_labels.size)
    num_bad_actions = 0
    for i in range(len(actions)):
        if abs(actions[i] - true_labels[i]) == 2:
            num_bad_actions += 1
    return num_bad_actions


if __name__ == '__main__':
    raw_data, num_labels = generate_data(save_txt=False, num_labels=3)
    overall_rewards = []
    for i in range(num_trials):
        overall_rewards.append(LinUCB(raw_data, num_labels, shuffle=True, alpha=0.))
    print(overall_rewards)
