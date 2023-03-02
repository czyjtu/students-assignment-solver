import random
from pydantic import BaseModel, Field, validator
from typing import NamedTuple
from ortools.sat.python import cp_model
import itertools as it
from collections import defaultdict

MAX_POINTS_PER_STUDENT = 9
MAX_STUDENTS_PER_TOPIC = 2

Vote = NamedTuple("Vote", topic_id=int, preference=int)


class Preferences(BaseModel):
    students: dict[str, list[Vote]]
    topics_count: int

    @validator("students", each_item=True)
    def check_sum(cls, val: list[Vote]):
        assert sum(v.preference for v in val) == MAX_POINTS_PER_STUDENT
        return val


def build_model(preferences: Preferences):
    topics = list(range(preferences.topics_count))

    model = cp_model.CpModel()

    student2thesis = {
        student: {topic: model.NewBoolVar(f"{student}-{topic}") for topic in topics}
        for student in preferences.students
    }

    topic2students = defaultdict(dict)
    for student, topics in student2thesis.items():
        for topic, var in topics.items():
            topic2students[topic][student] = var

    student2joy = {
        student: model.NewIntVar(0, MAX_POINTS_PER_STUDENT, f"{student}_joy")
        for student in preferences.students
    }

    # each student has exactly one topic
    for student, topics in student2thesis.items():
        model.AddExactlyOne(topics.values())

    # each topic can be done by at most MAX_STUDENTS_PER_TOPIC
    for topic, students in topic2students.items():
        model.Add(sum(students.values()) <= MAX_STUDENTS_PER_TOPIC)

    # consider preferneces
    for student, votes in preferences.students.items():
        topic2pref = {v.topic_id: v.preference for v in votes}
        for topic in topics:
            topic_given = student2thesis[student][topic]
            topic_value = topic2pref[topic] if topic in topic2pref else 0
            model.Add(student2joy[student] == topic_value).OnlyEnforceIf(topic_given)

    # maximize overall satisfaction
    model.Maximize(sum(student2joy.values()))
    return model, student2thesis


def solve_model(
    model: cp_model.CpModel, student2topic: dict[str, dict[int, cp_model.IntVar]]
) -> dict[str, int]:
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    solution = {}
    if status != cp_model.OPTIMAL:
        raise ValueError("Optimal solution not found")

    for student, topics in student2topic.items():
        for topic, var in topics.items():
            if solver.BooleanValue(var):
                solution[student] = topic

    return solution


def main():
    preferences = {
        "student1": [[1, 7], [2, 1], [3, 1]],
        "studnet2": [(1, 6), (3, 2), (4, 1)],
        "student3": [(1, 9)],
        "student_debil": [(i, 1) for i in range(MAX_POINTS_PER_STUDENT)],
    }
    model, vars = build_model(
        Preferences(**{"students": preferences, "topics_count": 20})
    )
    result = solve_model(model, vars)
    print(result)


def test_on_random():
    N_STUDENTS = 150
    N_TOPICS = 90

    students = {
        f"student_{i}": _random_preferences(MAX_POINTS_PER_STUDENT, N_TOPICS)
        for i in range(N_STUDENTS)
    }
    preferences = Preferences(students=students, topics_count=N_TOPICS)

    model, vars = build_model(preferences)
    result = solve_model(model, vars)
    print(result)


def _random_preferences(max_sum: int, max_topics: int) -> list[Vote]:
    points = [1] * max_sum
    indexes = [0] + sorted(random.choices(list(range(max_sum)), k=2)) + [max_sum]
    topics = random.sample(list(range(max_topics)), 3)

    preferences = [
        Vote(topic_id=t, preference=sum(points[i:j]))
        for t, (i, j) in zip(topics, zip(indexes[:-1], indexes[1:]))
    ]
    return preferences


if __name__ == "__main__":
    # test_on_random()
    main()
