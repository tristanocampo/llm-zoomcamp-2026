import requests
from minsearch import Index

'''
FUNCTIONS:
- load_faq_data(): Load FAQ data from the specified URL and return it as a list of documents. Each document is a dictionary containing the question, section, answer, and course.
- build_index(documents): Build an index from the loaded FAQ data. The index is created using the
'''

def load_faq_data():
    """
    Load FAQ data from the specified URL and return it as a list of documents.
    Each document is a dictionary containing the question, section, answer, and course.
    """

    docs_url = "https://datatalks.club/faq/json/courses.json"
    response = requests.get(docs_url)
    print(f"Status code: {response.status_code}")
    courses_raw = response.json()

    documents = []
    url_prefix = "https://datatalks.club/faq"

    for course in courses_raw:
        course_url = f"""{url_prefix}{course["path"]}"""

        course_response = requests.get(course_url)
        course_response.raise_for_status()
        course_data = course_response.json()

        documents.extend(course_data)
    return documents

def build_index(documents):
    """
    Build an index from the loaded FAQ data.
    The index is created using the MinSearch library, with specified text and keyword fields.
    """

    index = Index(
      text_fields = ['question', 'section', 'answer'],
      keyword_fields = ['course']
    )

    index.fit(documents)  
    
    return index

