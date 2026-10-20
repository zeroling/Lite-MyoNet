"""Current checkpoint-compatible teacher and student architectures."""
from .teacher import MyoFormerTeacher
from .student import MyoFormerStudent
MyoFormer = MyoFormerTeacher
LiteMyoNet = MyoFormerStudent
__all__ = ["MyoFormerTeacher", "MyoFormerStudent", "MyoFormer", "LiteMyoNet"]
