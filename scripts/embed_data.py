from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer


# Load the model
MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
sentence_transformer_model = SentenceTransformer(MODEL_NAME)

def create_segments(episode_data):
    window = 6  # number of lines to combine
    stride = 3  # number of lines to stride over, used to create overlap

    lines = episode_data['lines']

    transcript_segments = []
    for i in tqdm(range(0, len(lines), stride)):
        i_end = min(i + window, len(lines)-1)
        text = ' '.join(line['text'] for line in lines[i:i_end+1])

        start = lines[i]['timestamp']
        end = lines[i_end]['timestamp']
        
        transcript_segments.append({
            'start_time': start,
            'end_time': end,
            'episode_title': episode_data['episode_title'],
            'episode_number': episode_data['episode_number'],
            'text': text,
            'id': f"{episode_data['episode_number'].replace(' ', '_')}-{i}-{i_end}"
        })

    return transcript_segments


def embed_segments(transcript_segments):
    # Embed the segment text
    embeddings = sentence_transformer_model.encode([segment['text'] for segment in transcript_segments], show_progress_bar=True)
    for i, embedding in enumerate(embeddings):
        transcript_segments[i]['vector'] = embedding.tolist()

    return transcript_segments