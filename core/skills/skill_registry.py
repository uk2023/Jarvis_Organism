class SkillRegistry:
    """Registry of executable learned/native skills."""

    def __init__(self):
        self.skills = {}

    def register(self, name, handler):
        self.skills[name] = handler

    def get(self, name):
        return self.skills.get(name)
