import streamlit as st
import pandas as pd
import math

# Set page to wide layout
st.set_page_config(layout="wide")

# Basic initialization of session state
if 'init' not in st.session_state:
    st.session_state.init = True
    st.session_state.scores_dict = {}
    st.session_state.current_page = 0  # Renamed to be more clear
    st.session_state.saved_qs = set()
    st.session_state.all_scores = []

# Load data function with error handling
@st.cache_data
def load_data():
    try:
        return pd.read_csv("Language_bias_sentiment_analysis.csv")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
df = load_data()

# Basic validation
if df.empty:
    st.warning("No data available. Please check your CSV file.")
    st.stop()

if "question_id" not in df.columns:
    st.error("CSV file must contain a 'question_id' column")
    st.stop()

# Process data
df["question_id"] = df["question_id"].astype(str)

# Title
st.title("LLM Response Scoring App")

# Download section in sidebar
with st.sidebar:
    st.header("Download Scores")
    
    # Convert the stored scores to a DataFrame
    if st.session_state.all_scores:
        scores_df = pd.DataFrame(st.session_state.all_scores)
        
        # CSV download option
        csv_data = scores_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name="llm_scoring_results.csv",
            mime="text/csv"
        )
    else:
        st.info("Score some responses to enable downloads")

# Define the interval pattern
interval = 5
responses_per_page = 4  # Display 4 responses per page

# Calculate the total pages needed
total_rows = len(df)
# Calculate how many groups of pattern rows we need
# For example, with interval=5, each pattern group has rows like [0,5,10,15], [1,6,11,16], etc.
pattern_groups = interval
# Calculate how many pattern groups can be completely filled
filled_groups = total_rows // interval
# Calculate how many partial groups remain (for the remainder rows)
remaining_rows = total_rows % interval
# If we have remaining rows, we need one more group for them
total_pattern_sets = filled_groups + (1 if remaining_rows > 0 else 0)
# Total pages is the product of pattern groups and sets
total_pages = pattern_groups * total_pattern_sets if total_pattern_sets > 0 else pattern_groups

# Make sure page index is valid
if st.session_state.current_page >= total_pages:
    st.session_state.current_page = 0

# Calculate the base pattern index and the pattern offset
pattern_idx = st.session_state.current_page % pattern_groups  # 0-4 (which row pattern to show)
pattern_set = st.session_state.current_page // pattern_groups  # Which set of patterns we're in (0, 1, 2, etc.)

# Progress display
st.progress(st.session_state.current_page / max(1, total_pages - 1))
st.write(f"Page {st.session_state.current_page + 1} of {total_pages}")

# Create columns for responses
cols = st.columns(responses_per_page)

# Display responses with the interval pattern
displayed_responses = 0
for col_idx in range(responses_per_page):
    # For each column, we calculate which row to show
    # We start with our pattern index (0-4)
    # Then add the interval * pattern_set * responses_per_page to move to the correct pattern set
    # Then add interval * col_idx to move within the current pattern
    row_idx = pattern_idx + (pattern_set * interval * responses_per_page) + (col_idx * interval)
    
    # Check if we've reached the end of the dataframe
    if row_idx >= total_rows:
        break
    
    # Get the row data
    row = df.iloc[row_idx]
    question_id = row['question_id']
    model_name = row['llm']
    resp_key = f"response_{question_id}_{model_name}"
    score_key = f"score_{question_id}_{model_name}"
    
    with cols[col_idx]:
        st.subheader(f"Response {row_idx + 1}")
        st.write(f"Question ID: {question_id}")
        st.write(f"Question: {row['question_text']}")
        st.write(f"Model: {model_name}")
        st.text_area("Response", value=row["response"], height=400, key=resp_key)
        
        # Rating widget
        if score_key not in st.session_state:
            st.session_state[score_key] = 3  # Default score
            
        score = st.select_slider(
            "Rate response (1=Poor, 5=Excellent)",
            options=[1, 2, 3, 4, 5],
            value=st.session_state[score_key],
            key=f"slider_{score_key}"
        )
        
        # Store score in separate session state variable
        st.session_state[score_key] = score
        
        # Also store in our dictionary for saving later
        if question_id not in st.session_state.scores_dict:
            st.session_state.scores_dict[question_id] = {}
            
        st.session_state.scores_dict[question_id][model_name] = {
            "question_id": question_id,
            "llm": model_name,
            "score": score
        }
        
        displayed_responses += 1

if displayed_responses == 0:
    st.warning("No responses to display on this page.")

st.write("---")

# Navigation and save buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.current_page > 0:
        if st.button("⬅️ Previous Page"):
            st.session_state.current_page -= 1
            st.rerun()

with col2:
    if displayed_responses > 0:
        if st.button("💾 Save All Scores on Page"):
            # Get scores for current page
            scores_to_save = []
            for col_idx in range(responses_per_page):
                row_idx = pattern_idx + (pattern_set * interval * responses_per_page) + (col_idx * interval)
                
                if row_idx >= total_rows:
                    break
                    
                row = df.iloc[row_idx]
                question_id = row['question_id']
                model_name = row['llm']
                
                if question_id in st.session_state.scores_dict and model_name in st.session_state.scores_dict[question_id]:
                    data = st.session_state.scores_dict[question_id][model_name]
                    scores_to_save.append(data)
                    
                    # Add to all_scores list if not already saved
                    if question_id not in st.session_state.saved_qs:
                        st.session_state.all_scores.append(data)
                    # If already saved, update the existing entry
                    else:
                        # Find and update the existing entry
                        for i, item in enumerate(st.session_state.all_scores):
                            if item["question_id"] == question_id and item["llm"] == model_name:
                                st.session_state.all_scores[i] = data
                                break
                    
                    # Mark as saved
                    st.session_state.saved_qs.add(question_id)
            
            if scores_to_save:
                st.success(f"Saved {len(scores_to_save)} scores!")
            else:
                st.error("No scores to save.")

with col3:
    if st.session_state.current_page < total_pages - 1:
        if st.button("➡️ Next Page"):
            st.session_state.current_page += 1
            st.rerun()

# Save all button
if st.button("💾 Save All Remaining Scores"):
    # Collect all unsaved scores
    newly_saved = 0
    for qid in st.session_state.scores_dict:
        if qid not in st.session_state.saved_qs:
            for model, data in st.session_state.scores_dict[qid].items():
                st.session_state.all_scores.append(data)
                newly_saved += 1
            st.session_state.saved_qs.add(qid)
    
    if newly_saved > 0:
        st.success(f"Saved {newly_saved} scores!")
    else:
        st.info("No unsaved scores to save.")