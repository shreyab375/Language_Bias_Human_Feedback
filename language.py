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
    st.session_state.saved_pages = set()  # Track saved pages instead of questions
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

# Store current page indices for saving later
current_page_indices = []

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
    
    # Add this index to our current page indices
    current_page_indices.append(row_idx)
    
    # Get the row data
    row = df.iloc[row_idx]
    question_id = row['question_id']
    model_name = row['llm']
    resp_key = f"response_{row_idx}_{question_id}_{model_name}"
    score_key = f"score_{row_idx}_{question_id}_{model_name}"
    
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
        if row_idx not in st.session_state.scores_dict:
            st.session_state.scores_dict[row_idx] = {}
            
        st.session_state.scores_dict[row_idx] = {
            "row_index": row_idx,
            "question_id": question_id,
            "llm": model_name,
            "score": score,
            "page": st.session_state.current_page
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
    # Check if this page has already been saved
    current_page_already_saved = st.session_state.current_page in st.session_state.saved_pages
    
    save_button_label = "💾 Update Scores on Page" if current_page_already_saved else "💾 Save All Scores on Page"
    
    if displayed_responses > 0:
        if st.button(save_button_label):
            # Get scores for current page
            scores_to_save = []
            
            for row_idx in current_page_indices:
                if row_idx in st.session_state.scores_dict:
                    data = st.session_state.scores_dict[row_idx]
                    
                    # Check if this specific row index has already been saved
                    row_already_saved = False
                    for i, item in enumerate(st.session_state.all_scores):
                        if "row_index" in item and item["row_index"] == row_idx:
                            # Update the existing entry
                            st.session_state.all_scores[i] = data
                            row_already_saved = True
                            break
                    
                    # If not already saved, add it as new
                    if not row_already_saved:
                        st.session_state.all_scores.append(data)
                    
                    scores_to_save.append(data)
            
            # Mark this page as saved
            st.session_state.saved_pages.add(st.session_state.current_page)
            
            if scores_to_save:
                if current_page_already_saved:
                    st.success(f"Updated {len(scores_to_save)} scores!")
                else:
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
    # Collect all scores from all pages that haven't been explicitly saved
    newly_saved = 0
    current_unsaved_pages = set(range(total_pages)) - st.session_state.saved_pages
    
    # Check all row indices in scores_dict
    for row_idx, data in st.session_state.scores_dict.items():
        # If the page this row belongs to hasn't been saved yet
        if data["page"] in current_unsaved_pages:
            # Check if this specific row has already been saved
            row_already_saved = False
            for i, item in enumerate(st.session_state.all_scores):
                if "row_index" in item and item["row_index"] == row_idx:
                    # Update the existing entry
                    st.session_state.all_scores[i] = data
                    row_already_saved = True
                    break
            
            # If not already saved, add it as new
            if not row_already_saved:
                st.session_state.all_scores.append(data)
                newly_saved += 1
            
            # Mark the page as saved
            st.session_state.saved_pages.add(data["page"])
    
    if newly_saved > 0:
        st.success(f"Saved {newly_saved} scores!")
    else:
        st.info("No unsaved scores to save.")