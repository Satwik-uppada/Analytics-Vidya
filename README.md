# Smart Search Engine 🔍

A powerful and intelligent course search engine built with Streamlit and Elasticsearch that helps users find relevant online courses through advanced search algorithms and natural language processing.

![Image](https://github.com/user-attachments/assets/2aaeb798-810b-4cf3-ac39-2584495aed15)

![Smart Search Engine](https://github.com/user-attachments/assets/2aaeb798-810b-4cf3-ac39-2584495aed15)

## 🌟 Features

- **Smart Search** with fuzzy matching and spell correction
- **Real-time Results** with highlighted matches
- **Interactive UI** with expandable course details
- **Intelligent Ranking** based on multiple criteria
- **Debug Mode** for technical insights

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Search Engine:** Elasticsearch (8.15.3)
- **Language:** Python (3.12)
- **Text Processing:** NLTK, TextBlob
- **Data Management:** Pandas

## 📦 Dependencies

```python
streamlit
elasticsearch
pandas
nltk
textblob
```

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/Satwik-uppada/Analytics-Vidya.git
cd Analytics-Vidya
```

2. **Install requirements**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
- You will need to create an account on the [Elastic search website](https://www.elastic.co/elasticsearch).
- Then create a new project (I chose 14 days free trial)
- Give Index Name (I named it as search tool).
  
![Image](https://github.com/user-attachments/assets/b05e4092-b6b4-4cf0-a2a3-1fc9b6e73473)

![Image](https://github.com/user-attachments/assets/88723ef6-4cef-4aee-b9bb-4cb024653e19)

- You will get the elastic search URL and API key from here
### Change the URL and API key in the code
```bash
ELASTICSEARCH_URL=your_elasticsearch_url 
ELASTICSEARCH_API_KEY=your_api_key
```

4. **Run the application**
```bash
streamlit run app.py
```

## 🎯 Core Features Explained

### 1. Intelligent Search
- Implements fuzzy matching for typo tolerance
- Uses TextBlob for spell correction
- Preserves technical terms (ML, AI, NLP, etc.)
- Supports phrase matching

### 2. Result Ranking Algorithm
Prioritizes matches based on:
1. Exact phrase matches in course names (Boost: 10)
2. Partial matches in course names (Boost: 5)
3. Description matches (Boost: 3)
4. Curriculum matches (Boost: 1.5)
5. URL relevance (Boost: 0.5)

### 3. UI Components
- Clean, intuitive search interface
- Dynamic result highlighting
- Expandable curriculum sections
- Debug mode toggle
- Helpful search tips sidebar

## 💡 Search Tips

- Use specific keywords related to your interest
- Try different combinations of words
- Technical terms (ML, AI, Python, etc.) are preserved
- Results are ranked by relevance
- Expand curriculum details for more information

## 🔍 Debug Mode

Enable debug mode to view:
- Search relevance scores
- Course URLs
- Total vs. unique matches
- Error messages (if any)


## 🔒 Security

- Elasticsearch connection secured with API key
- HTML sanitization for safe rendering
- Error handling for API failures

## 🔄 Search Data Flow Diagram

```mermaid
graph LR
    A[User Input] --> B[Spell Correction]
    B --> C[Query Processing]
    C --> D[Elasticsearch Query]
    D --> E[Result Processing]
    E --> F[Highlighting]
    F --> G[UI Rendering]
```
## Images and UI Screenshots
![Image](https://github.com/user-attachments/assets/7c4dbf68-9ea8-4c3c-b983-d5f3b78d4cb6)


![Image](https://github.com/user-attachments/assets/052e9b89-4edc-4126-8a0a-9cc1ac0acc49)
Search for python courses. So python is highlighted

![Image](https://github.com/user-attachments/assets/9a7d974a-0f7a-4209-90cf-2a70a53e0fac)
Search for machi learni --> Textblob corrected the spelling to Machine Learning --> machine learning courses are displayed with highlighted correct spelling.

![Image](https://github.com/user-attachments/assets/ee58b577-2036-4e55-a51b-ff79e743c68a)
Debug mode activated. We can see the Accuracy source of relevant courses

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## 👥 Authors

- Satwik Uppada - *- A passionate data scientist and developer specializing in AI and machine learning.* - [Github Link](https://github.com/Satwik-uppada)

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- Elasticsearch for powerful search capabilities
- NLTK and TextBlob for text processing

---
Built with ❤️ using Streamlit and Elasticsearch
