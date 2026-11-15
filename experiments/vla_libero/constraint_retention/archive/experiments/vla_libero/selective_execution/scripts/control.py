"""Stop decisions consume only images, model replies and robot proprioception."""
from protocol import CHECK_INTERVAL, VOTE_TTL, released


class StopRule:
    def __init__(self, mode, threshold=150):
        self.mode = mode
        self.threshold = threshold
        self.streak = 0
        self.last_vote = None

    def eligible(self, step, gripper, last_action):
        if not released(gripper, last_action):
            self.streak = 0
            self.last_vote = None
            return False
        return step > 0 and step % CHECK_INTERVAL == 0

    def decide(self, step, reply=None):
        if self.mode in ["timer150", "timer_calibrated"]:
            done = step >= self.threshold
        elif self.mode in ["visual", "frozen"]:
            done = bool(reply and reply.get("ok") and reply.get("done") is True)
        else:
            done = False
        if self.last_vote is not None and step-self.last_vote > VOTE_TTL:
            self.streak = 0
        self.streak = self.streak+1 if done else 0
        self.last_vote = step if done else None
        return dict(done=done, streak=self.streak, stop=self.streak >= 2)
