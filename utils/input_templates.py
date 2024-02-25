GENERATIVE_CHOICES_INPUT_TEMPLATE = """\
question: which description describes the word " {0} " best in the \
following context? descriptions:[  " {1} ",  or " {2} " ] context: {3}\
"""

GENERATIVE_INPUT_TEMPLATE = """\
question: which description describes the word " {0} " best in the \
following context? context: {1}\
"""

TEMPLATES = {
    "generative": GENERATIVE_INPUT_TEMPLATE,
    "generative_choices": GENERATIVE_CHOICES_INPUT_TEMPLATE,
    "multiple_choice": GENERATIVE_CHOICES_INPUT_TEMPLATE
}