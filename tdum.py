import re

text = """This is a dummy text and this can be \n multiple lines i.e., with new lines \n and many of the special characters. \n But all this is a English text having a good sentence formation. 

, ,"""

# Remove extra commas (single or multiple) with optional spaces
cleaned_text = re.sub(r'(,\s*)+$', '', text).strip()

print(cleaned_text)
