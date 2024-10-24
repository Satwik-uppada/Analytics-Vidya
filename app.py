import streamlit as st
from elasticsearch import Elasticsearch
import pandas as pd
import re
import random
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(layout="wide")

def init_es_client():
    return Elasticsearch(
        "https://my-elasticsearch-project-d00981.es.us-east-1.aws.elastic.cloud:443",
        api_key="UjFnVXVwSUJSWFpTdXd5bEw4eE86VUFvVlhhc1VSaldCN28ybWdwMWFmQQ=="
    )

def get_highlight_color():
    """Return a set of predefined vibrant colors for highlighting"""
    colors = [
        "#FF69B4",  # Hot Pink
        "#00CED1",  # Dark Turquoise
        "#32CD32",  # Lime Green
        "#FF7F50",  # Coral
        "#9370DB",  # Medium Purple
    ]
    return random.choice(colors)

def highlight_text(text, query_terms):
    """Highlight search terms in text using vibrant background colors"""
    if not text or not query_terms:
        return text
    
    # Create a unique color for each term
    term_colors = {term: get_highlight_color() for term in query_terms if term}
    
    highlighted = text
    # Sort terms by length (longest first) to handle overlapping matches
    for term in sorted(query_terms, key=len, reverse=True):
        if term:
            color = term_colors[term]
            # Use HTML for highlighting with both background color and bold text
            pattern = re.compile(f'({re.escape(term)})', re.IGNORECASE)
            highlighted = pattern.sub(
                f'<span style="color: {color}; padding: 0.1em 0.2em; border-radius: 2px; font-weight: bold;">\\1</span>',
                highlighted
            )
    
    # Wrap the result in markdown to render HTML
    return f'<div style="font-family: sans-serif;">{highlighted}</div>'

# List of valid technical terms and abbreviations that should not be corrected
VALID_TERMS = {"ml", "ai", "nlp", "dl", "python", "sql", 'gen'}

def correct_spelling(query):
    """Corrects spelling mistakes in the user query while preserving valid terms."""
    corrected_words = []
    for word in query.split():
        # Preserve valid terms (like "ml", "ai", etc.)
        if word.lower() in VALID_TERMS:
            corrected_words.append(word)
        else:
            corrected_word = str(TextBlob(word).correct())
            corrected_words.append(corrected_word)
    
    return ' '.join(corrected_words)


def search_courses(query_text, client, index_name="search-tool"):
    # Split query into terms for highlighting
    nltk.download('stopwords')  # Download stopwords if not already present
    stop_words = set(stopwords.words('english'))

    # 2. Lemmatize and remove stop words from the query
    lemmatizer = WordNetLemmatizer()
    query_terms = [term.strip() for term in query_text.split()]
    query_terms = [lemmatizer.lemmatize(term.lower()) for term in query_terms if term.lower() not in stop_words]

    
    search_query = {
        "size": 30,
        "query": {
            "bool": {
                "should": [
                    # Course Name matches (highest priority)
                    {
                        "match_phrase": {
                            "Course Name": {
                                "query": query_text,
                                "boost": 10  # Exact phrase matches in course name
                            }
                        }
                    },
                    {
                        "match": {
                            "Course Name": {
                                "query": query_text,
                                "boost": 5,  # Partial matches in course name
                                "fuzziness": "AUTO"
                            }
                        }
                    },
                    # Description matches (medium priority)
                    {
                        "match_phrase": {
                            "Description": {
                                "query": query_text,
                                "boost": 3  # Exact phrase matches in description
                            }
                        }
                    },
                    {
                        "match": {
                            "Description": {
                                "query": query_text,
                                "boost": 2,  # Partial matches in description
                                "fuzziness": "AUTO"
                            }
                        }
                    },
                    # Curriculum matches
                    {
                        "match": {
                            "Curriculum": {
                                "query": query_text,
                                "boost": 1.5,  # Matches in curriculum
                                "fuzziness": "AUTO"
                            }
                        }
                    },
                    # Course Link matches (lowest priority but still relevant)
                    {
                        "match": {
                            "Course Link": {
                                "query": query_text,
                                "boost": 0.5,  # Matches in URL can indicate relevance
                                "fuzziness": "AUTO"
                            }
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        },
        "_source": ["Course Name", "Description", "Course Link", "Curriculum"]
    }
    
    try:
        response = client.search(index=index_name, body=search_query)
        
        # Deduplicate results based on Course Link while keeping highest scoring version
        unique_results = {}
        for hit in response['hits']['hits']:
            url = hit['_source']['Course Link']
            if url not in unique_results or hit['_score'] > unique_results[url]['_score']:
                # Add highlighted text to source
                hit['_source']['highlighted_name'] = highlight_text(
                    hit['_source']['Course Name'],
                    query_terms
                )
                hit['_source']['highlighted_description'] = highlight_text(
                    hit['_source']['Description'][:300] + "..." if len(hit['_source']['Description']) > 300 else hit['_source']['Description'],
                    query_terms
                )
                if 'Curriculum' in hit['_source']:
                    hit['_source']['highlighted_curriculum'] = highlight_text(
                        hit['_source']['Curriculum'],
                        query_terms
                    )
                unique_results[url] = hit
        
        # Convert back to list and sort by score
        deduped_results = sorted(
            unique_results.values(),
            key=lambda x: x['_score'],
            reverse=True
        )
        
        return deduped_results[:10], len(response['hits']['hits']), None
        
    except Exception as e:
        return [], 0, str(e)

# Streamlit UI
st.title("Course Search Engine")
st.sidebar.image("./square_og_image.jpeg")

# Initialize Elasticsearch client
try:
    client = init_es_client()
except Exception as e:
    st.error("Failed to connect to Elasticsearch. Please try again later.")
    st.stop()

# Search input
query = st.text_input("Enter your search query:", placeholder="e.g., machine learning, python, data science")

# Debug mode toggle
debug_mode = st.sidebar.checkbox("Enable Debug Mode")

if query:
    corrected_query = correct_spelling(query)
    st.success(f"Searching for {corrected_query}")
    with st.spinner("Searching..."):
        results, total_hits, error = search_courses(corrected_query, client)
        
        if error:
            st.error(f"Search error: {error}")
            if debug_mode:
                st.exception(error)
        elif results:
            # st.subheader(f"Found {len(results)} unique courses (from {total_hits} total matches)")
            
            for hit in results:
                source = hit['_source']
                
                # Create a card-like container for each result
                with st.container():
                    st.markdown("---")
                    
                    # Display course name with highlights
                    st.markdown(f"### [{source['highlighted_name']}]({source['Course Link']})",unsafe_allow_html=True)
                    
                    # Display description with highlights
                    st.markdown(f"**Description:** {source['highlighted_description']}",unsafe_allow_html=True)
                    
                    # Display curriculum if available and has matches
                    if 'highlighted_curriculum' in source:
                        with st.expander(label ='Curriculum', expanded=False, icon='🔯'):
                            st.markdown(f"**Curriculum Highlights:** {source['highlighted_curriculum']}",unsafe_allow_html=True)
                    
                    # Debug information
                    if debug_mode:
                        st.markdown("**Debug Info:**")
                        st.markdown(f"- Search Score: {hit['_score']:.2f}",unsafe_allow_html=True)
                        st.markdown(f"- URL: {source['Course Link']}",unsafe_allow_html=True)
                    
                    # View course button
                    st.markdown(f"[View Course Details]({source['Course Link']})",unsafe_allow_html=True)
        else:
            st.info("No courses found matching your search query. Try different keywords!")

# Sidebar with search tips
with st.sidebar:
    st.markdown("### Search Tips")
    st.markdown("""
    - Use specific keywords related to the course you're looking for
    - Try different combinations of words if you don't find what you need
    - Results are ranked by:
        1. Exact matches in course names (highest priority)
        2. Partial matches in course names
        3. Matches in description
        4. Matches in curriculum
        5. Relevant terms in course URL
    """)
    
    if debug_mode:
        st.markdown("### Debug Information")
        st.markdown("""
        When enabled, shows:
        - Search relevance scores
        - Course URLs
        - Total vs. unique matches
        - Any errors that occur
        """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Elasticsearch")