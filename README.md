# AI Product Research Agent

A Streamlit web app that automates product research and competitive analysis using AI. Analyze app reviews, changelogs, and blogs to identify pain points and generate actionable insights.

## Features

- **Automated Data Collection**: Fetch App Store reviews via RSS, scrape changelogs and blogs
- **AI-Powered Analysis**: Embed text using OpenAI, cluster pain points, and enable RAG-based Q&A
- **Evidence-Backed Insights**: Every insight is linked to source URLs and sample text
- **Export Capabilities**: Download pain points and documents as CSV or JSON
- **Cost-Efficient**: Embedding caching and efficient batching keep API costs low

## Demo

Create a session by entering:
- A research goal (e.g., "Identify top pain points in onboarding for note-taking apps")
- Competitor names
- Optional App Store/Play Store IDs
- Optional changelog/blog URLs
- Time window (e.g., last 90 days)

The agent will automatically:
1. Fetch app reviews and changelog content
2. Chunk and embed all text
3. Analyze pain point clusters
4. Store everything in your Supabase database

Then explore your insights:
- View clustered pain points with frequency and sentiment
- Ask natural language questions with RAG
- Export data for further analysis

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python (requests, psycopg2, beautifulsoup4, numpy, sklearn)
- **Database**: Supabase Postgres
- **AI**: OpenAI Embeddings (text-embedding-3-small) + GPT-4o-mini for RAG

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-product-research-agent.git
cd ai-product-research-agent
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Supabase

1. Create a free account at [supabase.com](https://supabase.com)
2. Create a new project
3. Go to **Project Settings > Database** to get your connection details
4. The app will automatically create the required tables on first run

### 5. Get OpenAI API Key

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create a new API key

### 6. Configure Environment Variables

**For local development:**

```bash
cp .env.example .env
# Edit .env with your credentials
```

**For Streamlit Cloud:**

Add secrets in the Streamlit Cloud dashboard under **Settings > Secrets**:

```toml
OPENAI_API_KEY = "sk-..."
SUPABASE_HOST = "your-project.supabase.co"
SUPABASE_PASSWORD = "your-password"
SUPABASE_USER = "postgres"
SUPABASE_DATABASE = "postgres"
SUPABASE_PORT = "5432"
```

### 7. Run Locally

```bash
streamlit run Home.py
```

## Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set `Home.py` as the main file
5. Add your secrets in the Secrets management section
6. Deploy!

## Cost Estimation

For a typical demo run (~500 reviews + 10 changelog pages):
- **OpenAI Embeddings**: ~$0.01 (text-embedding-3-small is very affordable)
- **OpenAI Chat**: ~$0.05 per question (GPT-4o-mini)
- **Supabase**: Free tier is sufficient

**Total: Less than $1 per demo session**

## Database Schema

```
sessions       - Research session metadata
documents      - Fetched content (reviews, changelogs)
vectors        - Text chunks with embeddings (JSON arrays)
insights       - Analyzed pain points with evidence
cache          - Embedding cache to minimize API costs
```

## Finding App IDs

### App Store (iOS)
- Go to the app page on apps.apple.com
- The ID is the number after `/id` in the URL
- Example: `https://apps.apple.com/us/app/notion/id1232780281` → ID: `1232780281`

### Play Store (Android)
- Go to the app page on play.google.com
- The ID is the `id` parameter in the URL
- Example: `https://play.google.com/store/apps/details?id=notion.id` → ID: `notion.id`

*Note: Play Store reviews require the official API for full access. The app will note this limitation.*

## Limitations

- Play Store reviews require Google Play Developer API (not included)
- Respects robots.txt and Terms of Service
- Some changelog sites may not be scrapeable
- Free tier Supabase has row limits

## License

MIT License - feel free to use and modify for your own projects.

## Contributing

Contributions welcome! Please open an issue or PR.
