"""Navigation stations and the verified skill library for the mobile kitchen scene.

Each skill carries the exact pi0.5 prompt and its visual acceptance contract:
ordered milestones, each judged on the current agentview image only, with earlier
confirmed milestones passed as text memory. `expect` maps answer keys to the values
that satisfy the milestone; `confirm` is the number of consecutive satisfied checks.
"""

STATIONS = {
    "kitchen_counter": "the work table with the white drawer cabinet, reachable by docking in front of it",
}

GRIPPER = "The white object with two fingers is the robot gripper, not a drawer or a bowl."

SKILLS = {
    "bowl_to_bottom_drawer_and_close": {
        "prompt": "put the black bowl in the bottom drawer of the cabinet and close it",
        "completion_questions": [
            {"question": "Is a bowl visible on the table top or in an open drawer?", "expected": False},
            {"question": "Are all drawers of the white cabinet fully closed?", "expected": True},
        ],
        "milestones": [
            {
                "name": "bowl_in_drawer",
                "fact": "the bowl was inside the pulled-out bottom drawer",
                "question": f"Look at the bowl and at the dark cabinet on the left. {GRIPPER} First describe in your own words, in one sentence, exactly where the bowl is in this image.",
                "answers": {
                    "bowl_on_table": "is the bowl standing on the table top?",
                    "bowl_in_drawer": "is the bowl inside a pulled-out drawer of the cabinet?",
                },
                "expect": {"bowl_on_table": False, "bowl_in_drawer": True},
                "confirm": 1,
            },
            {
                "name": "drawer_closed",
                "fact": "the bottom drawer was closed",
                "question": f"Look at the dark cabinet on the left side of the image. {GRIPPER} First describe the front of the cabinet in one sentence.",
                "answers": {"drawer_pulled_out": "is the bottom drawer pulled out?"},
                "expect": {"drawer_pulled_out": False},
                "confirm": 3,
            },
        ],
    },
}
