# FinSight Project Visual Template Brief

## Image Concept: "FinSight Agent in Action"

### **Overall Aesthetic**
- **Style**: Modern, tech-forward, minimal dark theme with blue accents
- **Color Palette**: 
  - Dark background: `#0f1115` (near-black)
  - Accent blue: `#4f8cff` (bright, contrasting)
  - Purple accent: `#7b5cff` (gradient element)
  - Text: `#e8eaed` (light gray)
  - Success green: `#2ecc71`

### **Layout: Left-to-Right Flow**

#### **LEFT SECTION: Chat Interface** (40% width)
- **Sidebar** (dark navy, `#0c0e12`)
  - Logo: "F" in gradient box (blue → purple)
  - Title: "FinSight" in clean sans-serif
  - "+ New Chat" button (outlined, hover state)
  - Chat list with 3 previous conversations:
    - "Apple vs Microsoft" — 2h ago
    - "Tesla stock analysis" — 5h ago
    - "Bitcoin today" — 1d ago
  - Scroll indicator showing more conversations below

- **Main Chat Area**
  - Input field at top: "Ask about a company or ticker, e.g. Apple or AAPL"
  - Research button (blue, `#4f8cff`)
  - Chat bubbles showing:
    - **User query**: "Give me a report on Apple" (right-aligned, blue bubble `#2b5cff`)
    - **Agent response** (left-aligned, dark panel):
      - Badge: "📊 FULL REPORT" (uppercase, small, blue accent)
      - Metrics grid (4 cards): Price, Market Cap, P/E Ratio, Sector
      - Sentiment tally: "▲ 5 Bullish | ▼ 2 Bearish | 3 Neutral" (colored chips)
      - Report preview (truncated markdown):
        ```
        ### Summary
        Apple shows strong growth momentum...
        
        ### Key Metrics
        - Trading at $195.50 USD
        - Market Cap: $3.05T
        ```

#### **MIDDLE SECTION: Architecture Diagram** (35% width)
- **Title**: "How FinSight Works" (top, white text)
- **Flow diagram** (vertical, top-to-bottom with arrows):
  ```
  [User Query] 
       ↓
  [route_query] → Intent Classification (Quick / Full / Out-of-Scope)
       ↓
  [resolve_ticker] → Extract ticker symbol
       ↓
  [fetch_market_data] → yfinance API
       ↓ (parallel split)
  ┌─[fetch_news]─────────────────────┐
  │ Tavily Search API                 │
  └─[fetch_sentiment]────────────────┘
       ↓ (parallel merge)
  [generate_report / generate_quick_answer]
       ↓
  [Response to User]
  ```

- **Legend icons** (small, aligned right):
  - 📊 Market Data
  - 📰 News Source
  - 💬 Social Sentiment
  - 🤖 LLM (Groq)
  - 💾 Memory (LangGraph)

#### **RIGHT SECTION: Tech Stack & Features** (25% width)

**Tech Stack** (stacked boxes with icons):
- 🐍 Python (98.6%)
- ⚡ FastAPI
- 🦜 LangChain + LangGraph
- 🧠 Groq LLM (llama-3.3-70b)
- 📊 yfinance
- 🔍 Tavily Search
- 💬 StockTwits API
- 🐳 Docker

**Key Features** (bullet list, blue accent dots):
- ✓ Multi-turn conversational memory
- ✓ Real-time market data
- ✓ Sentiment analysis from Twitter-like sources
- ✓ Intent classification (quick answer vs. full report)
- ✓ Anti-prompt-injection guards
- ✓ Out-of-scope detection
- ✓ Responsive chat UI
- ✓ Session-based persistence

---

## **Design Details**

### **Typography**
- Font: "Segoe UI" or "Inter" (clean, system font)
- Heading: 24px, bold, white
- Subheading: 14px, semi-bold, accent blue
- Body: 13px, regular, light gray
- Code/monospace: "Courier New" or "JetBrains Mono"

### **Component Styling**
- All boxes: rounded corners (8-14px border-radius)
- Buttons: 10px border-radius, shadow on hover (0 4px 12px rgba(0,0,0,0.3))
- Bubbles: fade-in animation (opacity 0→1 + slight translateY)
- Borders: `#2a2f3a` (dark gray, subtle)

### **Animated Elements** (if video/gif):
- Chat bubble slides in from left/right
- Typing dots animation in status bubble
- Flow diagram arrows animate top-to-bottom
- Metrics cards fade in sequentially

### **Dimensions**
- Suggested canvas: **1600px × 900px** (16:9 widescreen, optimal for GitHub README)
- Or: **1200px × 800px** (slightly tighter)
- Or: **2000px × 1200px** (high-res for presentations)

---

## **Usage Context**
This image will appear:
- **Top of README.md** — first thing visitors see
- **GitHub Discussions** — showcase what FinSight does
- **Portfolio** — demonstrate UI/UX + architecture integration
- **Presentations/Blog** — explain the project flow visually

---

## **Tool Recommendations for Creation**
1. **Figma** (fastest; free tier works)
   - Use component library for consistent buttons/cards
   - Export as PNG/SVG
   - Template: start with dark theme preset

2. **Adobe XD / Illustrator**
   - Vector-based precision
   - Professional gradient control

3. **Excalidraw** (quick & free)
   - Great for diagrams
   - Sketchy, whiteboard aesthetic (not ideal here, but functional)

4. **Python + Matplotlib/Plotly**
   - Script-based (great for automation)
   - Can embed real data / live updates

5. **Canvas.io / Canva** (DIY, beginner-friendly)
   - Drag-and-drop templates
   - Pre-made dark themes

---

## **Specific Visual Elements to Highlight**

### **Chat UI Mockup (Left)**
- Show a real example query that's interesting: "Give me a report on Tesla"
- Display both the metrics grid and sentiment badges (proof of multi-source aggregation)
- Show 2-3 past conversations in sidebar (proof of memory + persistence)

### **Architecture Diagram (Center)**
- Use consistent shape + color language:
  - Query step = circular badge
  - Processing node = rounded rectangle
  - API call = smaller rectangle with icon
  - Decision point = diamond shape (Optional)
- Arrows should be **thick, blue** (`#4f8cff`)
- Add subtle glow/shadow to important nodes (route_query, generate_report)

### **Tech Stack (Right)**
- Stack logos vertically with version/emphasis text
- Show "Free tier" badges (Groq, Tavily, yfinance are all free)
- Highlight the "LLM + Agents" story (not just a simple API wrapper)

---

## **Narrative / Message**
The image should communicate:
1. **"This is a real, working chat interface"** — not just a concept
2. **"Multi-step intelligence"** — not a simple prompt-response
3. **"Production-grade"** — FastAPI, LangGraph, structured state
4. **"Data-driven"** — real market, news, sentiment data flows together
5. **"Easy to deploy"** — Docker, modular, template-able

---

## **Optional: Create Multiple Variants**

### **Variant A: "The Happy Path"**
- Success state: query resolved, rich report rendered, sentiment visible
- Green checkmarks on successful API calls
- Upbeat energy

### **Variant B: "The Architecture"**
- Focus entirely on the flow diagram
- Minimal UI mockup
- More technical audience

### **Variant C: "Before & After"**
- Left: User's plain-text question "What's happening with Apple stock?"
- Right: Rich report with metrics, news, sentiment
- Arrow between them showing the transformation

---

Would you like me to create a specific version of this image, or do you have a preference for which tool to use?
